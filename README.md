# DBGroup-RAG-Tutorial

在使用pip下载完所有依赖，配置api_key后，使用`python Sample_rag.py`运行。

如果正常运行可以看到运行信息如：

`INFO:     Started server process [3967680]`

`INFO:     Waiting for application startup.`

`INFO:     Application startup complete.`

`INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

此时，可以访问该端口，如在命令行使用：

```Plain
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"query": "你的问题内容"}'
```

如果遇到端口冲突，修改代码中的端口即可。

