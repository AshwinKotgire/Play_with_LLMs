docker build -t basic_llm_container .
docker run -p 8000:8000 --name image_for_llm basic_llm_container