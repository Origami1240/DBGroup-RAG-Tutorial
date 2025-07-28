import os
import json
import requests
import time
from typing import List, Dict
from datetime import datetime

PARATERA_API_KEY = "your-api-key"
PARATERA_LLM_MODEL = "DeepSeek-R1-0528"
PARATERA_EMBEDDING_MODEL = "GLM-Embedding-3"
PARATERA_BASE_URL = "https://llmapi.paratera.com/v1"

os.environ["OPENAI_API_KEY"] = PARATERA_API_KEY
os.environ["OPENAI_BASE_URL"] = PARATERA_BASE_URL

# RAG 客户端类，用于向 RAG 服务发送请求
class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        # 设置默认超时时间
        self.default_timeout = (30, 600)  # (连接超时, 读取超时)
    
    def get_rag_response(self, query: str, max_retries: int = 3) -> Dict:
        """获取 RAG 响应，返回包含 answer 和 context 的字典"""
        for attempt in range(max_retries):
            try:
                print(f"尝试请求 (第{attempt + 1}次): {query[:50]}...")
                
                response = self.session.post(
                    f"{self.base_url}/ask",
                    json={"query": query},
                    timeout=self.default_timeout
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"请求成功，响应长度: {len(str(result))}")
                return {
                    "answer": result.get("answer", ""),
                    "context": result.get("context", [])
                }
                
            except requests.exceptions.Timeout as e:
                print(f"请求超时 (第{attempt + 1}次): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 递增等待时间
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"所有重试都失败，放弃请求")
                    return {"answer": "", "context": []}
                    
            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 (第{attempt + 1}次): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"连接失败，放弃请求")
                    return {"answer": "", "context": []}
                    
            except Exception as e:
                print(f"获取 RAG 响应时出错 (第{attempt + 1}次): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    return {"answer": "", "context": []}
        
        return {"answer": "", "context": []}

# 加载测试数据
def load_test_data(file_path: str) -> List[Dict]:
    """从 JSONL 文件加载测试数据"""
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        data = json.loads(line)
                        dataset.append(data)
                    except json.JSONDecodeError as e:
                        print(f"跳过无效的 JSON 行: {e}")
                        continue
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []
    
    print(f"成功加载 {len(dataset)} 条测试数据")
    return dataset

# 加载 URL 列表
def load_url_list(file_path: str) -> List[Dict]:
    """从文件加载 URL 列表，新格式：姓名 URL"""
    url_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        url = parts[1]
                        url_data.append({
                            "name": name,
                            "url": url
                        })
                    else:
                        print(f"跳过格式不正确的行: {line}")
    except Exception as e:
        print(f"读取 URL 列表时出错: {e}")
        return []
    
    print(f"成功加载 {len(url_data)} 个 URL")
    return url_data

# 测试单个 RAG 服务
def test_rag_service(url: str, name: str, test_data: List[Dict], output_dir: str) -> Dict:
    """测试单个 RAG 服务"""
    print(f"\n正在测试 RAG 服务: {name} ({url})")
    
    # 初始化 RAG 客户端
    rag = RAGClient(url)
    
    # 创建评估数据集
    dataset = []
    
    print("正在处理测试用例...")
    for i, item in enumerate(test_data):
        query = item["user_input"]
        reference = item["reference"]
        
        print(f"处理测试用例 {i+1}/{len(test_data)}: {query[:50]}...")
        
        # 获取 RAG 响应
        rag_response = rag.get_rag_response(query)
        response = rag_response["answer"]
        relevant_docs = rag_response["context"]
        
        dataset.append({
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference
        })
        time.sleep(5)
    
    # 保存测试结果
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    result_file = os.path.join(output_dir, f"test_results_{safe_name}.json")
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"测试结果已保存到 {result_file}")
    
    # 使用 ragas 进行评估
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness
        from langchain_openai import ChatOpenAI
        
        print("正在使用 RAGAS 进行评估...")
        
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        # 初始化评估用的 LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=PARATERA_LLM_MODEL,
                api_key=PARATERA_API_KEY,
                base_url=PARATERA_BASE_URL,
            )
        )
        
        # 执行评估
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
            llm=evaluator_llm
        )
        
        print(f"=== RAGAS 评估结果 ({name}) ===")
        print(result)
        
        # 保存评估结果
        evaluation_file = os.path.join(output_dir, f"evaluation_results_{safe_name}.json")
        try:
            if hasattr(result, 'dict'):
                result_dict = result.dict()
            elif hasattr(result, '__dict__'):
                result_dict = result.__dict__
            else:
                result_str = str(result)
                result_dict = {"result": result_str}
            
            with open(evaluation_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            print(f"评估结果已保存到 {evaluation_file}")
        except Exception as save_error:
            print(f"保存评估结果时出错: {save_error}")
            # 保存为文本格式
            txt_file = os.path.join(output_dir, f"evaluation_results_{safe_name}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(str(result))
            print(f"评估结果已保存到 {txt_file}")
        
        return {
            "url": url,
            "name": name,
            "test_results_file": result_file,
            "evaluation_results_file": evaluation_file,
            "evaluation_result": str(result)
        }
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return {
            "url": url,
            "name": name,
            "test_results_file": result_file,
            "error": str(e)
        }

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"rag_evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"结果将保存到目录: {output_dir}")
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_data = load_test_data("rag_testset.jsonl")
    
    if len(test_data) == 0:
        print("没有加载到测试数据，退出")
        return
    
    # 加载 URL 列表
    print("正在加载 URL 列表...")
    urls = load_url_list("url_lists.txt")
    
    if len(urls) == 0:
        print("没有加载到 URL，退出")
        return
    
    # 测试每个 URL
    results = []
    for i, url_item in enumerate(urls):
        name = url_item["name"]
        url = url_item["url"]
        print(f"\n{'='*50}")
        print(f"测试进度: {i+1}/{len(urls)}")
        print(f"姓名: {name}")
        print(f"URL: {url}")
        print(f"{'='*50}")
        
        try:
            result = test_rag_service(url, name, test_data, output_dir)
            results.append(result)
        except Exception as e:
            print(f"测试 {name} ({url}) 时出错: {e}")
            results.append({
                "name": name,
                "url": url,
                "error": str(e)
            })
        
        # 添加延迟，避免请求过于频繁
        if i < len(urls) - 1:
            print("等待 5 秒后继续下一个测试...")
            time.sleep(5)
    
    # 保存总体结果摘要
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_urls": len(urls),
            "successful_tests": len([r for r in results if "error" not in r]),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print("所有测试完成！")
    print(f"结果保存在: {output_dir}")
    print(f"摘要文件: {summary_file}")
    print(f"成功测试: {len([r for r in results if 'error' not in r])}/{len(urls)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
