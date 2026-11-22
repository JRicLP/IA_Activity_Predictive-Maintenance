# Sentinela · backend + front

API local (e tela) de apoio à inspeção de máquinas. A Random Forest foi treinada no
AI4I 2020 sintético, com 200 árvores, profundidade máxima 10, sem pesos de
classe e sem reamostragem. O alerta é emitido quando o score é maior que 0,50.
O score descreve associação com a classe de falha nas condições informadas.
Uma recomendação de inspeção serve como apoio à análise.

## Executar

Use Python 3.13.9. A partir da raiz do repositório:

```powershell
cd backend
py -3.13 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Em Linux ou macOS, use `../.venv/bin/python` no lugar de
`..\.venv\Scripts\python.exe`.

A API também serve o front (`../front/static`). Com o servidor no ar, abra
[http://127.0.0.1:8000](http://127.0.0.1:8000) para a tela e
[a documentação interativa](http://127.0.0.1:8000/docs) para testar a API.

A aba "Validação do modelo" lê `models/metricas.json`, gerado pelo
`train_export.py`. Se o arquivo não existir, rode o treinamento (seção abaixo)
uma vez; sem ele, `GET /metricas` responde 503.

O modelo já está em `models/random_forest.joblib`. O CSV é necessário apenas
para repetir o treinamento. Versione o arquivo do modelo e
`models/metadata.json` juntos. Use artefatos Joblib da sua equipe e mantenha
as versões de dependências indicadas para o modelo salvo.

## Contrato com o frontend

- `GET /health` retorna o estado da API e a versão do modelo carregado.
- `GET /metricas` retorna as métricas, gráficos e exemplos da validação (`models/metricas.json`).
- `POST /predict` recebe JSON. Use `Content-Type: application/json`.

```json
{
  "type": "L",
  "air_temperature_k": 301.9,
  "process_temperature_k": 310.4,
  "rotational_speed_rpm": 1286,
  "torque_nm": 62.8,
  "tool_wear_min": 22
}
```

A API calcula diferença de temperatura, potência e desgaste × torque a
partir das medições. O campo `type` aceita `L`, `M` ou `H`. Envie
valores numéricos JSON nas unidades indicadas. Campos faltando ou inválidos
retornam HTTP 422. Para simular esse erro, troque `type` por `"X"` no
exemplo.

O exemplo acima retorna classe 1 e score de aproximadamente 0,873. Para
um caso sem alerta, use `type="L"`, temperaturas de 297,4 K e 308,6 K,
rotação de 1.459 rpm, torque de 41,3 Nm e desgaste de 196 min. O retorno
esperado é classe 0, com score de aproximadamente 0,0013.

| Campo da resposta | Significado |
| --- | --- |
| `prediction` | Classe 0 ou 1 |
| `model_score` | Score de falha entre 0 e 1. Exiba como score, sem interpretar como probabilidade calibrada de quebra futura |
| `inspection_recommended` | Indicação de inspeção, verdadeira quando o score supera o limiar |
| `message` | Texto para exibir ao usuário |
| `threshold` | Limiar usado |
| `model_version` | Identificador da versão do modelo |
| `temp_diff`, `power_w`, `wear_x_torque` | Features calculadas pela API a partir das medições (K, W e min·Nm). A tela mostra esses indicadores |
| `out_of_training_range` | Lista de medições fora dos mínimos e máximos vistos no treino. A API ainda calcula a previsão, que deve ser interpretada com cautela |

Por padrão, o CORS aceita frontends locais nas portas 3000 e 5173,
com `localhost` ou `127.0.0.1`. Para outro endereço, configure
`SENTINELA_ALLOWED_ORIGINS` com a lista de origens separadas por vírgula.

## Explicabilidade e relatório

- `POST /explain` calcula contribuições locais da classe positiva 1 e devolve
  uma importância global cacheada.
- `POST /dashboard/start` inicia o dashboard Shapash interativo.
- `POST /reports/generate` recalcula a previsão no servidor e gera um relatório
  estruturado por Gemini a partir de evidências JSON.

Configure o Gemini a partir de `.env.example`. A chave permanece no backend.
Consulte `docs/EXPLICABILIDADE_E_RELATORIOS.md` para limites e execução.

## Reproduzir o treinamento

Use o CSV [AI4I 2020 da UCI](https://archive.ics.uci.edu/dataset/601/ai4i),
com o mesmo conteúdo e ordem de linhas usados no notebook. Na pasta
`backend`, execute (o CSV fica na raiz do repositório):

```powershell
..\.venv\Scripts\python.exe train_export.py --dataset ..\ai4i2020.csv
```

Se o CSV estiver em `data/`, passe `--dataset ..\data\ai4i2020.csv`.
O script reproduz a divisão original e usa os 6.000 registros de treino.
Além do modelo e do `metadata.json`, ele gera o `models/metricas.json` com as
métricas da validação (o teste não é lido). Versione os três arquivos juntos.
Os metadados registram a configuração, as versões e os hashes do CSV e do
modelo. A exportação foi comparada com os 2.000 registros de validação
originais, reproduzindo as classes e os scores de referência. O teste
reservado fica fora do treinamento e dessa verificação.
