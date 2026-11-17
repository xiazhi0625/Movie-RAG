import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import chromadb
import pandas as pd


from sentence_transformers import SentenceTransformer
_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class MovieRetriever:
    def __init__(self, db_path="./chroma_movie_db_v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="movie_collection_v2")

    def hybrid_search(self, query_text, n_results=3, weight_semantic=0.7, weight_rating=0.3):
        """核心：粗排 + 权重融合精排"""
        if not query_text or not query_text.strip():
            return "", []

        try:
            query_vec = _model.encode(query_text).tolist()
            # 1. 粗排：扩大池子捞 50 个
            results = self.collection.query(query_embeddings=[query_vec], n_results=50)

            if not results['ids'][0]:
                return "", []

            # 2. 提取数据计算
            candidates = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                # 平滑映射到 (0, 1] 区间防负数
                similarity = 1.0 / (1.0 + distance)
                rating = meta.get('rating', 0.0)

                candidates.append({
                    'title': meta.get('title', '未知'),
                    'genres': meta.get('genres', '未知'),
                    'rating': rating,
                    'similarity': similarity,
                    'metas': meta # 保留原始元数据
                })

            df = pd.DataFrame(candidates)

            # 3. Min-Max 归一化
            if df['similarity'].max() != df['similarity'].min():
                df['norm_sim'] = (df['similarity'] - df['similarity'].min()) / (df['similarity'].max() - df['similarity'].min())
            else:
                df['norm_sim'] = 0.5

            if df['rating'].max() != df['rating'].min():
                df['norm_rat'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
            else:
                df['norm_rat'] = 0.5

            # 4. 权重融合打分
            df['final_score'] = df['norm_sim'] * weight_semantic + df['norm_rat'] * weight_rating
            df = df.sort_values(by='final_score', ascending=False)

            # 5. 提取 Top_k
            top_df = df.head(n_results)
            context_text = ""
            metas_list = []
            for _, row in top_df.iterrows():
                context_text += f"- {row['title']} (类型: {row['genres']}, 评分: {row['rating']})\n"
                metas_list.append(row['metas'])

            return context_text, metas_list
        except Exception as e:
            print(f"⚠️ 检索层报错: {e}")
            return "", []
