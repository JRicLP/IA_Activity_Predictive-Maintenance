# Sentinela: apoio à inspeção industrial (AI4I 2020)

## Como rodar
1. Crie o ambiente e instale:
   python -m venv .venv
   .venv\Scripts\activate        (Windows)   |   source .venv/bin/activate  (Linux/Mac)
   pip install -r requirements.txt
2. Garantir que o `ai4i2020.csv` esteja nesta pasta (mesmo arquivo do notebook EDA).
3. Treine o modelo (gera modelo_rf.joblib e metricas.json):
   python treino.py
4. Suba o servidor:
   uvicorn main:app --reload
5. Abra http://127.0.0.1:8000

Use o mesmo ambiente (venv) para treinar e para rodar a API, senão o joblib pode reclamar de versão do scikit-learn.
Docs automáticas da API: http://127.0.0.1:8000/docs
