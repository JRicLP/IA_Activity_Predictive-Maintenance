from pathlib import Path


INDEX_PATH = Path(__file__).resolve().parents[2] / "front" / "static" / "index.html"


def frontend_html() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


def test_frontend_has_user_facing_error_messages_and_loading_guard():
    html = frontend_html()

    assert "Não consegui conectar à API. Confirme que o servidor uvicorn está rodando." in html
    assert "Valores fora da faixa aceita (confira unidades em K, rpm, Nm e min)." in html
    assert "A API respondeu com erro. Veja o terminal do uvicorn." in html
    assert "Failed to fetch" not in html
    assert "if(analiseEmAndamento)return false" in html
    assert '$("go").setAttribute("aria-busy",String(ativo))' in html
    assert "finally{" in html


def test_frontend_uses_dynamic_validation_summary_and_translated_importance_labels():
    html = frontend_html()

    assert "total=tn+fp+fn+tp,falhas=fn+tp" in html
    assert "100/falhas" in html
    assert "Resultados no conjunto de validação (2.000 linhas)." not in html
    assert "Só 68 falhas na validação" not in html
    assert 'air_temperature_k:"Temperatura do ar"' in html
    assert 'wear_x_torque:"Desgaste × torque"' in html


def test_frontend_exposes_training_ranges_and_disables_agent_by_default():
    html = frontend_html()

    for reference_range in (
        "295,3–304,5 K",
        "305,7–313,8 K",
        "1.168–2.886 rpm",
        "3,8–76,6 Nm",
        "0–253 min",
    ):
        assert f"Faixa observada no treino: {reference_range}" in html

    assert "const AGENTE_HABILITADO=false" in html
    assert "[hidden]{display:none!important}" in html
    assert '<link rel="icon" href="data:,">' in html
