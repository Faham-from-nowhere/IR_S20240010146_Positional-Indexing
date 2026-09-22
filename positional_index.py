import os
import re
import time
import json
import tracemalloc
import psutil
import statistics
from collections import defaultdict

# 1. COMPRESSION PIPELINE (Gap & Variable-Byte)

def vb_encode_number(n):
    bytes_list = []
    while True:
        bytes_list.insert(0, n % 128)
        if n < 128: break
        n //= 128
    bytes_list[-1] += 128
    return bytes_list

def vb_decode(byte_data):
    numbers, n = [], 0
    for byte in byte_data:
        if byte < 128: n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers

def serialize_postings(doc_dict):
    flat_list = [len(doc_dict)]
    sorted_docs = sorted(doc_dict.keys())
    last_doc = 0
    for doc_id in sorted_docs:
        flat_list.append(doc_id - last_doc) 
        last_doc = doc_id
        positions = doc_dict[doc_id]
        flat_list.append(len(positions))
        last_pos = 0
        for pos in positions:
            flat_list.append(pos - last_pos)
            last_pos = pos
    vb_bytes = bytearray()
    for num in flat_list:
        vb_bytes.extend(vb_encode_number(num))
    return bytes(vb_bytes)

def deserialize_postings(vb_bytes):
    numbers = vb_decode(vb_bytes)
    if not numbers: return {}
    doc_dict = {}
    num_docs = numbers[0]
    idx = 1
    last_doc = 0
    for _ in range(num_docs):
        doc_id = last_doc + numbers[idx]
        last_doc = doc_id
        idx += 1
        num_pos = numbers[idx]
        idx += 1
        positions, last_pos = [], 0
        for _ in range(num_pos):
            pos = last_pos + numbers[idx]
            last_pos = pos
            positions.append(pos)
            idx += 1
        doc_dict[doc_id] = positions
    return doc_dict

# 2. POSITIONAL & N-GRAM INDEXER

class PositionalIndexer:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))
        self.lexicon = {}
        
        self.total_docs = 0
        self.raw_integer_count = 0 
        self.unigram_tokens = 0
        self.bigram_tokens = 0
        self.trigram_tokens = 0

    def add_document(self, doc_id, text):
        self.total_docs += 1
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        
        for i in range(len(tokens)):
            # Unigram
            self.index[tokens[i]][doc_id].append(i)
            self.raw_integer_count += 2
            self.unigram_tokens += 1
            
            # Bigram
            if i < len(tokens) - 1:
                bigram = f"{tokens[i]} {tokens[i+1]}"
                self.index[bigram][doc_id].append(i)
                self.raw_integer_count += 2
                self.bigram_tokens += 1
                
            # Trigram
            if i < len(tokens) - 2:
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                self.index[trigram][doc_id].append(i)
                self.raw_integer_count += 2
                self.trigram_tokens += 1

    def flush_to_disk(self, index_dir="index_data"):
        os.makedirs(index_dir, exist_ok=True)
        postings_path = os.path.join(index_dir, "postings.dat")
        
        current_offset = 0
        with open(postings_path, 'wb') as f:
            for term, doc_dict in self.index.items():
                compressed_bytes = serialize_postings(doc_dict)
                length = len(compressed_bytes)
                f.write(compressed_bytes)
                self.lexicon[term] = {"offset": current_offset, "length": length}
                current_offset += length
                
        with open(os.path.join(index_dir, "lexicon.json"), 'w') as f:
            json.dump(self.lexicon, f)

# 3. QUERY ENGINE

class QueryEngine:
    def __init__(self, index_dir="index_data"):
        self.lexicon = json.load(open(os.path.join(index_dir, "lexicon.json"), 'r'))
        self.f_postings = open(os.path.join(index_dir, "postings.dat"), 'rb')
            
    def fetch_postings(self, term):
        if term not in self.lexicon: return {}, 0, 0, 0
        meta = self.lexicon[term]
        
        t0 = time.perf_counter()
        self.f_postings.seek(meta["offset"])
        seek_ms = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        vb_bytes = self.f_postings.read(meta["length"])
        read_ms = (time.perf_counter() - t1) * 1000
        
        t2 = time.perf_counter()
        doc_dict = deserialize_postings(vb_bytes)
        decode_ms = (time.perf_counter() - t2) * 1000
        
        return doc_dict, seek_ms, read_ms, decode_ms

    def merge_postings(self, post1, post2, offset=1):
        merged = {}
        docs1, docs2 = sorted(list(post1.keys())), sorted(list(post2.keys()))
        i = j = 0
        
        t_start = time.perf_counter()
        while i < len(docs1) and j < len(docs2):
            d1, d2 = docs1[i], docs2[j]
            if d1 == d2:
                pos1, pos2 = post1[d1], post2[d2]
                valid_pos = []
                p1 = p2 = 0
                while p1 < len(pos1) and p2 < len(pos2):
                    if pos2[p2] == pos1[p1] + offset:
                        valid_pos.append(pos1[p1])  # Keep original start position
                        p1 += 1; p2 += 1
                    elif pos2[p2] <= pos1[p1]: p2 += 1
                    else: p1 += 1
                if valid_pos: merged[d1] = valid_pos
                i += 1; j += 1
            elif d1 < d2: i += 1
            else: j += 1
            
        merge_ms = (time.perf_counter() - t_start) * 1000
        return merged, merge_ms

    def phrase_query_unigram(self, phrase):
        tokens = phrase.lower().split()
        if not tokens: return {}
        
        current_docs, s1, r1, d1 = self.fetch_postings(tokens[0])
        total_seek, total_read, total_dec, total_merge = s1, r1, d1, 0
        
        for idx, token in enumerate(tokens[1:]):
            next_docs, s2, r2, d2 = self.fetch_postings(token)
            total_seek += s2; total_read += r2; total_dec += d2
            current_docs, m_time = self.merge_postings(current_docs, next_docs, offset=idx+1)
            total_merge += m_time
            if not current_docs: break
            
        return current_docs, total_seek, total_read, total_dec, total_merge
    
    def phrase_query_ngram(self, phrase):
        docs, s, r, d = self.fetch_postings(phrase.lower())
        return docs, s, r, d, 0 

# 4. FULL BENCHMARKING & ARTIFACT EXPORT

if __name__ == "__main__":
    import nltk
    nltk.download('reuters', quiet=True)
    from nltk.corpus import reuters
    
    print("Loading Reuters Corpus...")
    docs = [reuters.raw(fileid) for fileid in reuters.fileids()]
    
    indexer = PositionalIndexer()
    process = psutil.Process(os.getpid())
    
    print(f"Building N-Gram Positional Index for {len(docs)} documents... (This takes 1-2 mins)")
    rss_before = process.memory_info().rss
    tracemalloc.start()
    
    t0 = time.time()
    for i, text in enumerate(docs): 
        indexer.add_document(i, text)
    indexing_time = time.time() - t0
    
    _, peak_malloc = tracemalloc.get_traced_memory()
    rss_after = process.memory_info().rss
    tracemalloc.stop()
    
    print("Computing Vocabulary Stats...")
    vocab_keys = list(indexer.index.keys())
    vocab_total = len(vocab_keys)
    unigram_vocab = sum(1 for k in vocab_keys if k.count(' ') == 0)
    bigram_vocab = sum(1 for k in vocab_keys if k.count(' ') == 1)
    trigram_vocab = sum(1 for k in vocab_keys if k.count(' ') == 2)
    
    print("Flushing to disk (Compression & Serialization)...")
    indexer.flush_to_disk()
    indexer.index.clear() # Clear RAM
    
    # VALIDATION: MULTI-QUERY TESTS
    queries = [
        "stock", "market", "bank", "federal", "reserve", "interest", "rates", 
        "business", "corporate", "financial", "investment", "bonds", "equity", 
        "dividend", "yield", "japan", "tokyo", "london", "international", "growth",
        "wall street", "stock market", "interest rates", "federal reserve", 
        "united states", "new york", "dow jones", "balance sheet", 
        "economic growth", "central bank", "foreign exchange", "trade deficit",
        "prime rate", "bull market", "bear market", "money supply",
        "stock market crash", "federal reserve board", "united states dollar",
        "consumer price index", "wall street journal", "new york times"
    ]
    
    print(f"\nExecuting {len(queries)} query tests and compiling terminal metrics...")
    engine = QueryEngine()
    
    cold_times, warm_times, ngram_times = [], [], []
    
    with open("query_results.txt", "w", encoding="utf-8") as f_out:
        f_out.write("="*60 + "\nBENCHMARK QUERY RESULTS\n" + "="*60 + "\n\n")
        
        for q in queries:
            words = q.split()
            
            # Cold vs Warm Unigram
            res_cold, sk1, rd1, dc1, mg1 = engine.phrase_query_unigram(q)
            cold_total = sk1 + rd1 + dc1 + mg1
            cold_times.append(cold_total)
            
            res_warm, sk2, rd2, dc2, mg2 = engine.phrase_query_unigram(q)
            warm_total = sk2 + rd2 + dc2 + mg2
            warm_times.append(warm_total)
            
            f_out.write(f"QUERY: '{q}'\n")
            f_out.write(f"  Matches: {len(res_warm)} documents\n")
            f_out.write(f"  Cold Cache Latency: {cold_total:.3f} ms\n")
            f_out.write(f"  Warm Cache Latency: {warm_total:.3f} ms\n")
            
            if len(words) > 1:
                res_ngram, sk3, rd3, dc3, mg3 = engine.phrase_query_ngram(q)
                ngram_total = sk3 + rd3 + dc3 + mg3
                ngram_times.append(ngram_total)
                f_out.write(f"  N-Gram Lookup Latency: {ngram_total:.3f} ms\n")
                
                # Hard assert logic check
                assert res_warm == res_ngram, f"Mismatch on: {q}"
                
            if res_warm:
                f_out.write(f"  Top Document Hits:\n")
                for did in list(res_warm.keys())[:3]:
                    f_out.write(f"    - DocID {did}: Positions {res_warm[did]}\n")
            f_out.write("-" * 50 + "\n")
            

    # PRINT TERMINAL METRICS
    raw_int_bytes = indexer.raw_integer_count * 4
    postings_size = os.path.getsize("index_data/postings.dat")
    total_disk_size = postings_size + os.path.getsize("index_data/lexicon.json")
    
    print("\n" + "="*70)
    print("        SYSTEMS METRICS: POSITIONAL N-GRAM INDEX")
    print("="*70)
    
    print(f"\n[1] Indexing Throughput & Counts")
    print(f"  Total Indexing Time        : {indexing_time:.2f} seconds")
    print(f"  Throughput (Unigrams)      : {indexer.unigram_tokens / indexing_time:,.0f} tokens/sec")
    print(f"  Unigrams  (Tokens/Vocab)   : {indexer.unigram_tokens:,} / {unigram_vocab:,}")
    print(f"  Bigrams   (Tokens/Vocab)   : {indexer.bigram_tokens:,} / {bigram_vocab:,}")
    print(f"  Trigrams  (Tokens/Vocab)   : {indexer.trigram_tokens:,} / {trigram_vocab:,}")
    print(f"  Total Indexed Terms        : {vocab_total:,}")
    
    print(f"\n[2] Memory Tracking (In-Memory Build Phase)")
    print(f"  Peak Traced Python Alloc   : {peak_malloc / (1024*1024):.2f} MB")
    print(f"  Process RSS Growth         : {(rss_after - rss_before) / (1024*1024):.2f} MB")
    
    print(f"\n[3] Storage & Compression (Disk Footprint)")
    print(f"  Raw Integer Representation : {raw_int_bytes / (1024*1024):.2f} MB (32-bit)")
    print(f"  Compressed postings.dat    : {postings_size / (1024*1024):.2f} MB")
    print(f"  Integer Compression Ratio  : {raw_int_bytes / postings_size:.2f}x")
    print(f"  Total Index Size on Disk   : {total_disk_size / (1024*1024):.2f} MB")
    
    print(f"\n[4] Query Performance Averages ({len(queries)} Queries)")
    print(f"  Unigram Merge (COLD CACHE) : {statistics.mean(cold_times):.3f} ms / query")
    print(f"  Unigram Merge (WARM CACHE) : {statistics.mean(warm_times):.3f} ms / query")
    if ngram_times:
        print(f"  Direct N-Gram (WARM CACHE) : {statistics.mean(ngram_times):.3f} ms / query")
        print(f"  Performance Gain (N-Gram vs Merge): {statistics.mean(warm_times) / statistics.mean(ngram_times):.2f}x faster")
    print("="*70)

    # EXPORTING 7 ARTIFACT FILES (FULL VS TRIMMED)
    print("\nExporting artifacts to disk (This may take a minute)...")
    
    sorted_terms = sorted(engine.lexicon.keys())
    
    # 1 & 2. Vocabulary Files
    with open("vocabulary_full.txt", "w", encoding="utf-8") as f_v_full, \
         open("vocabulary_trimmed.txt", "w", encoding="utf-8") as f_v_trim:
             
        f_v_full.write(f"TOTAL UNIQUE TERMS (RAW REGEX): {len(sorted_terms)}\n" + "="*40 + "\n")
        
        trimmed_count = sum(1 for t in sorted_terms if not any(c.isdigit() for c in t))
        f_v_trim.write(f"TOTAL UNIQUE TERMS (ALPHABETICAL): {trimmed_count}\n")
        f_v_trim.write(f"NOISE REDUCTION: Removed {len(sorted_terms) - trimmed_count:,} numeric terms.\n")
        f_v_trim.write("="*40 + "\n")
        
        for term in sorted_terms:
            f_v_full.write(f"{term}\n")
            if not any(c.isdigit() for c in term):
                f_v_trim.write(f"{term}\n")

    # 3, 4, 5, & 6. Index Dump Files
    with open("index_data/postings.dat", "rb") as f_post, \
         open("index_unigram_full.txt", "w", encoding="utf-8") as f_uni_full, \
         open("index_ngram_full.txt", "w", encoding="utf-8") as f_ngram_full, \
         open("index_unigram_trimmed.txt", "w", encoding="utf-8") as f_uni_trim, \
         open("index_ngram_trimmed.txt", "w", encoding="utf-8") as f_ngram_trim:
             
        for count, term in enumerate(sorted_terms):
            if count > 0 and count % 200000 == 0:
                print(f"  ... exported {count:,} terms ...")
                
            meta = engine.lexicon[term]
            f_post.seek(meta["offset"])
            vb_bytes = f_post.read(meta["length"])
            postings = deserialize_postings(vb_bytes)
            
            # Determine format string
            output_str = f"TERM: '{term}' ({len(postings)} docs)\n"
            for doc_id, positions in postings.items():
                output_str += f"  DocID {doc_id}: {positions}\n"
            output_str += "\n"
            
            # Determine routing
            is_unigram = (term.count(' ') == 0)
            has_numbers = any(c.isdigit() for c in term)
            
            # Route to Full Indexes
            if is_unigram: f_uni_full.write(output_str)
            else: f_ngram_full.write(output_str)
                
            # Route to Trimmed Indexes (if no numbers)
            if not has_numbers:
                if is_unigram: f_uni_trim.write(output_str)
                else: f_ngram_trim.write(output_str)

    print("\n[PASS] All tasks complete! 7 Artifact files generated:")
    print("  1. query_results.txt (Latency benchmarks)")
    print("  2. vocabulary_full.txt (All terms)")
    print("  3. vocabulary_trimmed.txt (Alphabetical only)")
    print("  4. index_unigram_full.txt")
    print("  5. index_ngram_full.txt")
    print("  6. index_unigram_trimmed.txt")
    print("  7. index_ngram_trimmed.txt")