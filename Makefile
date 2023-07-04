zen:
	python -c "import this"

#UTILS
install_requirements :
	pip install -U pip wheel
	pip install -r requirements.txt

reset_models_logs :
	rm -rf models/*
	mkdir -p models

reset_data :
	rm -rf data/*
	mkdir -p data

# VM
copy_code:
	gcloud compute scp --recurse ../ $(USER)@$(INSTANCE):~/sarcasme

# TESTS
test_gcp :
	python test/test_bq_access.py

#API
run_api:
	uvicorn sarcasme.api.fast:app --reload
test_api:
	curl -X 'GET' \
  'http://127.0.0.1:8000/predict?sentence=ok' \
  -H 'accept: application/json'


#Docker
run_docker:
	docker run -e PORT=8000 -p 8000:8000 sarcasme:dev
build_docker:
	docker build --tag=$(GCR_IMAGE):dev .

build_docker_gcp:
	docker build -t $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(GCR_IMAGE) .

run_docker_gcp:
	docker run -e PORT=8000 -p 8080:8000 $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(GCR_IMAGE)
push_docker_gcr:
	docker push $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(GCR_IMAGE)

deploy:
	gcloud run deploy --image $(GCR_MULTI_REGION)/$(PROJECT_ID)/$(GCR_IMAGE) --region $(GCR_REGION) --memory 4Gi

test_api_gcp:
	curl -X 'GET' \
  'https://sarcasme-uwd4nnxq3a-ew.a.run.app/predict?sentence=ok' \
  -H 'accept: application/json'
