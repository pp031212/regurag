import configparser
import os
import re
import chromadb
import torch
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
from openai import OpenAI
from typing import List, Dict, Any



# 文档处理模块: 语义/规则边界层级切分 (Semantic Parent-Child Chunking)
class DocumentProcessor:
    def __init__(self, child_chunk_size=50):
        # 抛弃了 parent_chunk_size，父块大小现在由规则边界自然决定
        self.child_chunk_size = child_chunk_size

    def process(self, text):
        chunks_data = []

        # ==========================================
        # 1. 切分父块：按【语义/规则边界】精准下刀
        # ==========================================
        # 这里的正则 (?=...) 是“前向断言”，意思是：找到这些词的位置，在这里切开，但不要把这些词吃掉。
        # 匹配目标：中文数字+顿号(如"一、" "十、")，或者特定的总起句(如"扣分处理标准如下：")
        rule_boundary_pattern = r'(?=(?:[一二三四五六七八九十]+、)|(?:扣分处理标准如下：))'

        # 按照规则边界切分出原始的父块数组
        raw_parents = re.split(rule_boundary_pattern, text)

        # 过滤掉可能切出来的空字符串
        parent_chunks = [p.strip() for p in raw_parents if p.strip()]

        # ==========================================
        # 2. 为每个自然父块生成对应的子块
        # ==========================================
        for p_idx, p_text in enumerate(parent_chunks):
            # 在父块内部，按标点符号切分句子
            sentences = re.split(r'(?<=[。！？；\n])', p_text)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_child = []
            for s in sentences:
                # 组装子块，控制在 child_chunk_size 左右
                if sum(len(x) for x in current_child) + len(s) > self.child_chunk_size and current_child:
                    chunks_data.append({
                        "child_text": "".join(current_child),
                        "parent_text": p_text,  # 绑定完整的规则段落
                        "parent_id": f"rule_doc_{p_idx}"  # 标识符改个名，代表规则块
                    })
                    current_child = []
                current_child.append(s)

            # 收尾剩余的句子
            if current_child:
                chunks_data.append({
                    "child_text": "".join(current_child),
                    "parent_text": p_text,
                    "parent_id": f"rule_doc_{p_idx}"
                })

        # 打印一下切分报告，让你心里有数
        print(
            f"[*] 语义切分完毕！共切出 {len(parent_chunks)} 个核心规则大块 (Parent)，衍生出 {len(chunks_data)} 个检索子块 (Child)。")
        return chunks_data


# 向量检索模块 (粗排): BGE + L2 归一化
class VectorStore:
    # 使用专门做检索的 BGE 模型
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        # 新建一个集合，避免和之前的 BERT 向量混淆
        self.collection = self.client.get_or_create_collection(name="rag_test_bge")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    # 增加 is_query 参数，用来判断是不是用户的提问
    def _get_embedding(self, text, is_query=False):
        # BGE 的官方“咒语”：只在查询时添加，入库文档不加
        if is_query:
            text = "为这个句子生成表示以用于检索相关文章：" + text

        # 加上 max_length=512 ，防止长文本直接导致内存报错
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 直接提取 [CLS]（即第 0 个位置）的向量
        embeddings = outputs.last_hidden_state[:, 0]

        # L2 归一化
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings.squeeze().tolist()

    def _generate_id(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_documents(self, chunks_data):
        print(f"正在构建粗排向量库，共{len(chunks_data)}个文本块...")
        # 提取子块文本进行向量化
        embeddings = [self._get_embedding(item["child_text"]) for item in chunks_data]
        # ID 必须唯一，使用子块文本+父ID生成
        ids = [self._generate_id(item["child_text"] + item["parent_id"]) for item in chunks_data]
        # 将父块文本存入元数据
        metadatas = [{"parent_text":item["parent_text"],"parent_id": item["parent_id"]} for item in chunks_data]
        documents = [item["child_text"] for item in chunks_data]
        self.collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings,metadatas=metadatas)
        print("粗排库入库完成！")

    def search(self, query, top_k=5):  # 粗排阶段放宽名额，多捞一点
        # 检索时，明确告诉模型这是用户提问，触发 is_query=True
        query_vector = self._get_embedding(query, is_query=True)

        # 增加 include 参数，要求 ChromaDB 把底层向量一起返回
        # 在 include 里加上 "metadatas"
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "distances", "embeddings", "metadatas"]
        )

        structured_results = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]
            embs = results["embeddings"][0]  # 拿到检索出的文档向量
            metas = results["metadatas"][0]

            for doc, dist, doc_id, emb, meta in zip(docs, dists, ids, embs, metas):
                structured_results.append({
                    "id": doc_id,
                    "child_text": doc,
                    "parent_text": meta["parent_text"],
                    "parent_id": meta["parent_id"],
                    "distance": dist,
                    "embedding": emb  # 把向量存入字典，带出模块
                })

        # 同时返回查询向量和候选结果
        return query_vector, structured_results
# MMR 算法对候选文本块进行重排
def mmr_rerank(query_embedding, doc_embeddings, top_k=5, lambda_mult=0.5):
    """
    使用 MMR 算法对候选文本块进行重排。

    参数:
    query_embedding: 问题的向量表达，shape (1, hidden_size)
    doc_embeddings: 候选文本块的向量表达列表，shape (N, hidden_size)
    top_k: 最终需要返回的文本块数量
    lambda_mult: 多样性权重因子，0.5代表相关性与多样性五五开

    返回:
    selected_indices: 被选中的文本块索引列表
    """

    # 将所有的 doc_embeddings 转为 numpy 数组以便计算
    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # 1. 计算所有候选文档与 Query 的相似度 (公式前半部分)
    query_doc_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # 提取所有候选文档的初始索引
    unselected = list(range(len(doc_embeddings)))
    selected = []

    # 2. 选出第一篇文档：与 Query 相似度最高的那篇
    best_idx = np.argmax(query_doc_similarities)
    selected.append(best_idx)
    unselected.remove(best_idx)

    # 3. 循环选出剩余的 top_k - 1 篇文档
    while len(selected) < top_k and len(unselected) > 0:
        best_score = -np.inf
        best_idx_to_add = -1

        # 遍历所有还没被选中的文档
        for i in unselected:
            # 公式前半部分：当前文档与 Query 的相似度
            sim_to_query = query_doc_similarities[i]

            # 公式后半部分：当前文档与【已选文档】的最大相似度
            # 即评估它有多“重复”
            selected_docs_embeddings = doc_embeddings[selected]
            sim_to_selected = cosine_similarity([doc_embeddings[i]], selected_docs_embeddings)[0]
            max_sim_to_selected = np.max(sim_to_selected)

            # 计算 MMR 得分
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected

            # 更新最高得分和对应的索引
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx_to_add = i

        # 将得分最高的文档加入已选列表，并从待选列表中移除
        selected.append(best_idx_to_add)
        unselected.remove(best_idx_to_add)

    return selected
# 重排模块 (精排)
class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    def rerank(self, query, retrieved_docs, top_k=2):
        if not retrieved_docs:
            return []
        print(f"正在对粗排的{len(retrieved_docs)}个文本块进行交叉审阅...")
        pairs = [(query, doc["text"]) for doc in retrieved_docs]

        with torch.no_grad():
            inputs = self.tokenizer(pairs, return_tensors="pt", truncation=True, padding=True, max_length=512)
            scores = self.model(**inputs,return_dict=True).logits.view(-1,).float().tolist()
        for i in range(len(retrieved_docs)):
            retrieved_docs[i]["rerank_score"] = scores[i]

        ranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_docs[:top_k]
# 生成模块:使用LLM API
class LLMGenerator:
    def __init__(self, config_path="config.ini"):
        self.config = self._load_config(config_path)

        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url']
        )
        self.model_name = self.config['model_name']

    def _load_config(self, config_path):
        config = configparser.ConfigParser()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        config.read(config_path, encoding='utf-8')

        return {
            'api_key': config.get('llm', 'api_key'),
            'base_url': config.get('llm', 'base_url'),
            'model_name': config.get('llm', 'model_name')
        }

    def generate(self, query, context):
        # 升级版 Prompt：赋予大模型推导、计算和分情况讨论的权利
        prompt = f"""你是一个具备逻辑推理能力、且重视证据边界的专业问答助手。请严格基于【参考资料】回答【问题】。

        必须遵守以下规则：
        1. 回答只能依据参考资料，不得脱离参考资料自由发挥。
        2. 允许根据常识和上下文做合理推断，但必须明确标注这是“参照理解”或“相近条款解释”，不能说成资料已明确规定。
        3. 如果资料中同时存在局部行为条款和全局扣分处理标准，必须同时回答两者。
        4. 如果资料中已经出现全局扣分处理标准，禁止说“没有全局扣分处理标准”。
        5. 如果资料不足以支撑确定性结论，要明确写出“不确定的原因是什么”。
        6. 不要引入与当前问题主线无关的规则。
        7. 不要使用 LaTeX，只能使用纯文本。

        请严格按以下格式输出：

        【直接依据】
        - 写出参考资料中与问题最直接相关的条款内容

        【推理/判断】
        - 先说明这是直接适用，还是参照相近条款理解
        - 如果需要计算或判断区间，逐步写出过程
        - 如果存在信息不足或规则空档，明确指出

        【结论】
        - 给出最终结论
        - 如果结论带有推断性质，要明确写“这是基于相近条款的解释”

        【参考资料】：
        {context}

        【问题】：
        {query}
        """

        print("\n等待大模型思考并生成回答...")
        # 为了防止你还没配置好 API Key 导致代码报错中断，
        # 这里做了一个简单的 Try-Catch。填好真实 Key 后它就会真正请求云端。
        try:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": "你是一个严谨的 AI 助手。"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1  # 调低温度，让大模型回答更严谨，少发散
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"大模型 API 调用失败（请检查 API Key 和网络）。\n错误信息：{e}\n\n[模拟生成的后备回答]: 人工智能的研究领域主要包括机器人、语言识别等。"


class QueryRewriter:
    def __init__(self, config_path="config.ini"):
        self.config = self._load_config(config_path)

        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url']
        )
        self.model_name = self.config['model_name']

    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        config.read(config_path, encoding='utf-8')

        # 优先读取 [rewrite_llm]，如果没有配置 key/url，就 fallback 复用 [llm] 的
        api_key = config.get('rewrite_llm', 'api_key', fallback=config.get('llm', 'api_key'))
        base_url = config.get('rewrite_llm', 'base_url', fallback=config.get('llm', 'base_url'))

        # 模型名字必须明确指定
        model_name = config.get('rewrite_llm', 'model_name', fallback=config.get('llm', 'model_name'))

        return {
            'api_key': api_key,
            'base_url': base_url,
            'model_name': model_name
        }

    def rewrite(self, user_query):
        prompt = f"""你是一个专业的 RAG 搜索词优化专家。
    用户会输入一个口语化的问题，你需要把它改写成更适合检索“规章制度/扣分制度”文本的搜索关键词。

    【目标】
    让搜索词尽量贴近规章制度原文措辞，而不是扩展成泛化的校纪处分术语。

    【要求】
    1. 只输出关键词，用空格隔开。
    2. 不要输出任何解释、标点或废话。
    3. 优先补充规章制度里常见的原文表达，例如：
       扣分 处罚 处理标准 累计扣除 累计扣分 退学 无条件退学 学费不予退还
    4. 不要随意扩展成知识库里未必存在的泛化术语，例如：
       记过 留校察看 开除学籍 寻衅滋事 等，除非用户问题本身明确提到。
    5. 如果用户问题涉及“几次、后果、处罚、会怎样”这类问法，要优先补充“累计处理标准”相关词。
    6. 保留用户问题里的核心实体，不要替换掉。

    用户问题：{user_query}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            expanded_keywords = response.choices[0].message.content.strip()
            return expanded_keywords
        except Exception as e:
            print(f"[警告] Query Rewrite 失败: {e}，将降级使用原问题检索")
            return ""

# RAG编排器
class RAGPipeline:
    def __init__(self):
        self.processor = DocumentProcessor(child_chunk_size=50)
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.llm = LLMGenerator()
        self.rewriter = QueryRewriter()
    def ingest_data(self, document_text):
        chunks = self.processor.process(document_text)
        self.vector_store.add_documents(chunks)

    def ask(self, query, top_k_retrieve=20, top_k_mmr=8, top_k_rerank=5):
        # 0. 智能 Query Rewrite
        print("\n[*] 正在请求大模型进行搜索词重写...")
        expanded_keywords = self.rewriter.rewrite(query)

        search_query = query + " " + expanded_keywords if expanded_keywords else query
        print(f"[*] 动态 Query Rewrite 完成！\n最终检索词: {search_query}\n")

        # 1. 粗排：基于子块进行向量匹配
        query_vector, retrieved_docs = self.vector_store.search(search_query, top_k=top_k_retrieve)

        # 1.1 粗排结果严格去重
        seen_keys = set()
        deduped_docs = []
        for doc in retrieved_docs:
            unique_key = doc["child_text"] + doc["parent_text"]
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduped_docs.append(doc)

        print(f"[*] 粗排捞取 {len(retrieved_docs)} 个子块，精准去重后剩余 {len(deduped_docs)} 个")

        valid_docs = [doc for doc in deduped_docs if doc.get("embedding") is not None]
        doc_embeddings = [doc["embedding"] for doc in valid_docs]

        if not valid_docs:
            answer = self.llm.generate(query, "")
            return {
                "answer": answer,
                "raw_retrieved_docs": retrieved_docs,
                "deduped_docs": deduped_docs,
                "mmr_selected_docs": [],
                "reranked_parents": [],
                "final_context_parents": []
            }

        # 2. MMR 筛选：基于子块挤出水分
        selected_indices = mmr_rerank(
            query_vector,
            doc_embeddings,
            top_k=min(top_k_mmr, len(valid_docs)),
            lambda_mult=0.6  # 原来 0.8 偏高，容易为了“多样性”引入噪声
        )
        mmr_selected_docs = [valid_docs[i] for i in selected_indices]

        # 2.5 制度类问题：强制保留“全局处理标准”相关子块
        policy_trigger_words = ["次", "处罚", "后果", "会怎样", "会有什么样的处罚", "会怎么样"]
        must_keep_child_keywords = [
            "扣分处理标准如下",
            "累计扣除1—10分",
            "累计扣除11—20分",
            "累计扣除21—30分",
            "累计扣除40分及以上",
            "无条件退学",
            "学费不予退还"
        ]

        is_policy_question = any(w in query for w in policy_trigger_words)

        if is_policy_question:
            existing_keys = {
                doc["child_text"] + doc["parent_text"]
                for doc in mmr_selected_docs
            }

            forced_count = 0
            for doc in deduped_docs:
                unique_key = doc["child_text"] + doc["parent_text"]
                if unique_key in existing_keys:
                    continue

                text_to_check = doc["child_text"] + "\n" + doc["parent_text"]
                if any(k in text_to_check for k in must_keep_child_keywords):
                    mmr_selected_docs.append(doc)
                    existing_keys.add(unique_key)
                    forced_count += 1

            if forced_count > 0:
                print(f"[*] 为制度类问题强制保留了 {forced_count} 个全局处理标准子块")

        # 3. Small-to-Big：子块映射回父块
        unique_parents = {}
        for doc in mmr_selected_docs:
            pid = doc.get("parent_id")
            if pid not in unique_parents:
                unique_parents[pid] = {
                    "parent_id": pid,
                    "text": doc["parent_text"],
                    "matched_children": [doc["child_text"]]
                }
            else:
                unique_parents[pid]["matched_children"].append(doc["child_text"])

        parent_docs_for_rerank = list(unique_parents.values())

        if not parent_docs_for_rerank:
            answer = self.llm.generate(query, "")
            return {
                "answer": answer,
                "raw_retrieved_docs": retrieved_docs,
                "deduped_docs": deduped_docs,
                "mmr_selected_docs": mmr_selected_docs,
                "reranked_parents": [],
                "final_context_parents": []
            }

        # 4. 父块精排：这里要改成用 search_query，而不是原始 query
        reranked_parents = self.reranker.rerank(
            search_query,
            parent_docs_for_rerank,
            top_k=min(top_k_rerank, len(parent_docs_for_rerank))
        )

        # 5. 最终上下文组装：不能完全交给 rerank，必须保住关键父块
        must_keep_parent_keywords = [
            "扣分处理标准如下",
            "累计扣除1—10分",
            "累计扣除11—20分",
            "累计扣除21—30分",
            "累计扣除40分及以上",
            "无条件退学",
            "学费不予退还"
        ]

        low_value_parent_keywords = [
            "以下为和鸣教育管理制度",
            "处罚不是目的",
            "旨在帮助学员",
            "请同学们自觉遵守"
        ]

        def is_global_policy_parent(parent_doc):
            text = parent_doc["text"]
            return any(k in text for k in must_keep_parent_keywords)

        def is_low_value_parent(parent_doc):
            text = parent_doc["text"]
            return any(k in text for k in low_value_parent_keywords)

        final_context_parents = []
        added_parent_texts = set()

        # 5.1 先加入高价值 rerank 父块，跳过明显低价值父块
        for doc in reranked_parents:
            if is_low_value_parent(doc):
                continue
            if doc["text"] not in added_parent_texts:
                final_context_parents.append(doc)
                added_parent_texts.add(doc["text"])

        # 5.2 如果是制度处罚类问题，强制补回“全局处理标准父块”
        if is_policy_question:
            forced_parent_count = 0
            for parent_doc in parent_docs_for_rerank:
                if parent_doc["text"] in added_parent_texts:
                    continue
                if is_global_policy_parent(parent_doc):
                    final_context_parents.append(parent_doc)
                    added_parent_texts.add(parent_doc["text"])
                    forced_parent_count += 1

            if forced_parent_count > 0:
                print(f"[*] 最终上下文中强制补回了 {forced_parent_count} 个全局处理标准父块")

        # 5.3 如果前面被过滤得太少，再从 rerank 里补齐
        for doc in reranked_parents:
            if len(final_context_parents) >= top_k_rerank:
                break
            if doc["text"] not in added_parent_texts:
                final_context_parents.append(doc)
                added_parent_texts.add(doc["text"])

        # 5.4 控制最终上下文长度，避免过长
        # 可以保留 top_k_rerank + 1 或 +2，给强制保留留空间
        final_context_parents = final_context_parents[: top_k_rerank + 2]

        # 6. 拼接上下文
        context = "\n\n".join([doc["text"] for doc in final_context_parents])

        print(f"[*] 最终送入大模型的父块数: {len(final_context_parents)}")
        for i, doc in enumerate(final_context_parents, 1):
            preview = doc["text"].replace("\n", " ")[:120]
            print(f"    [{i}] {preview}...")

        # 7. 生成答案
        answer = self.llm.generate(query, context)

        return {
            "answer": answer,
            "raw_retrieved_docs": retrieved_docs,
            "deduped_docs": deduped_docs,
            "mmr_selected_docs": mmr_selected_docs,
            "reranked_parents": reranked_parents,
            "final_context_parents": final_context_parents
        }

# 解析llm返回的回答
def parse_llm_answer(result):
    print("=" * 40)
    print("最终回答: ")
    print(result.get("answer", ""))
    print("-" * 40)

    def _fmt_distance(x):
        if x is None:
            return "N/A"
        try:
            return f"{x:.2f}"
        except Exception:
            return str(x)

    def _fmt_score(x):
        if x is None:
            return "N/A"
        try:
            return f"{x:.2f}"
        except Exception:
            return str(x)

    raw_retrieved_docs = result.get("raw_retrieved_docs")
    if raw_retrieved_docs:
        print("原始粗排命中的【子块】(Raw Child Chunks):")
        for i, doc in enumerate(raw_retrieved_docs):
            print(f"[{i + 1}] (粗排距离: {_fmt_distance(doc.get('distance'))}) 子块: {doc.get('child_text', '')}")
        print("-" * 40)

    deduped_docs = result.get("deduped_docs")
    if deduped_docs:
        print("去重后的【子块】(Deduped Child Chunks):")
        for i, doc in enumerate(deduped_docs):
            print(f"[{i + 1}] (粗排距离: {_fmt_distance(doc.get('distance'))}) 子块: {doc.get('child_text', '')}")
        print("-" * 40)

    mmr_selected_docs = result.get("mmr_selected_docs")
    if mmr_selected_docs:
        print("MMR 选中的【子块】(MMR-selected Child Chunks):")
        for i, doc in enumerate(mmr_selected_docs):
            print(f"[{i + 1}] (粗排距离: {_fmt_distance(doc.get('distance'))}) 子块: {doc.get('child_text', '')}")
        print("-" * 40)

    reranked_parents = result.get("reranked_parents")
    if reranked_parents:
        print("最终喂给大模型的【父块】(Parent Chunks):")
        for i, doc in enumerate(reranked_parents):
            rerank_score = doc.get("rerank_score", None)
            matched_children = doc.get("matched_children", [])

            print(f"[{i + 1}] (裁判打分: {_fmt_score(rerank_score)})")

            if matched_children:
                print(f"包含命中的子块: {matched_children}")

            print(f"送入的完整父块原文: {doc.get('text', '')}")
            print()
    print("=" * 40)

if __name__ == '__main__':
    rag_system = RAGPipeline()

    #读取和鸣教育管理制度，写入到raw_document变量中
    with open("和鸣教育管理制度精简chunk版.md", "r", encoding="utf-8") as f:
        raw_document = f.read()

    rag_system.ingest_data(raw_document)

    user_query = "我如果在和鸣教育扣分累计31分会怎样？"


    result = rag_system.ask(user_query, top_k_retrieve=15, top_k_rerank=3)
    print(f"\n用户提问: {user_query}")
    parse_llm_answer(result)

