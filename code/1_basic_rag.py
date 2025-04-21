import lazyllm
from lazyllm import bind
"""数据流实现"""
# 文档加载
documents = lazyllm.Document(dataset_path="/mnt/lustre/share_data/lazyllm/data/rag_master/", embed=lazyllm.OnlineEmbeddingModule(source="qwen"), create_ui=False)
prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. \
      In this task, you need to provide your answers based on the given context and questions.'

# RAG 数据流
with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)   
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str = "".join([node.get_content() for node in nodes]),
            query = query,
        )
    ) | bind(query=ppl.input)           # ppl.retriever的检索结果会直接传到下一步（因为是pipeline数据流），也就是传给nodes，然后用bind参数输入了最开始的输入ppl.input(也就是用户的query)

    # 同理ppl.formatter得到的字典(就是原本的问题+query),会直接传给ppl.llm，等机遇上面的llm({dict})
    ppl.llm = lazyllm.OnlineChatModule(source='qwen', model="qwen-turbo").prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))  # 看下
    

ans = ppl("何为大学")
print(ans)