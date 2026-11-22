/* agente.js — espaço do módulo de explicabilidade (agente de IA com SHAP).
 *
 * O index.html chama iniciarAgente(raiz, obterContexto) na PRIMEIRA vez que o
 * painel abre (botão "Explicar com IA").
 *
 *  - raiz: o <div id="agente-root"> dentro do painel. Monte o chat aqui.
 *  - obterContexto(): devolve a última análise feita na tela, ou null se ainda
 *    não houve nenhuma. Chame de novo a cada pergunta, o usuário pode analisar
 *    outra medição com o painel aberto.
 *        { entrada:   corpo do POST /predict
 *                     (type, air_temperature_k, process_temperature_k,
 *                      rotational_speed_rpm, torque_nm, tool_wear_min),
 *          resultado: resposta do POST /predict
 *                     (model_score, threshold, inspection_recommended,
 *                      temp_diff, power_w, wear_x_torque, ...) }
 *
 * Para chamar o backend, use caminho relativo (mesma origem):
 *     fetch("explain", { method: "POST",
 *       headers: { "Content-Type": "application/json" },
 *       body: JSON.stringify(obterContexto().entrada) })
 * A chave da API do LLM fica só no backend, nunca aqui.
 *
 * Troque o conteúdo desta função pelo agente real.
 */
window.iniciarAgente = function (raiz, obterContexto) {
  raiz.replaceChildren();

  const estado = document.createElement("p");
  estado.className = "agente-status";
  estado.textContent = "Realize uma previsão para habilitar a explicação.";

  const acoes = document.createElement("div");
  acoes.className = "agente-actions";
  const explicar = document.createElement("button");
  explicar.type = "button";
  explicar.className = "btn";
  explicar.textContent = "Explicar previsão";
  const relatorio = document.createElement("button");
  relatorio.type = "button";
  relatorio.className = "btn2";
  relatorio.textContent = "Gerar relatório com Gemini";
  acoes.append(explicar, relatorio);

  const saida = document.createElement("div");
  raiz.append(acoes, estado, saida);

  function contextoAtual() {
    const contexto = obterContexto();
    if (!contexto) {
      estado.textContent = "Realize uma previsão antes de solicitar uma explicação.";
      return null;
    }
    return contexto;
  }

  function mensagemErro(status, fallback) {
    if (status === 503) return "O recurso não está disponível no momento. Confira a configuração do backend.";
    if (status === 429) return "Limite gratuito atingido ou cooldown ativo. Aguarde e tente novamente.";
    if (status === 504) return "A consulta excedeu o tempo limite.";
    if (status === 502) return "O provedor retornou uma falha ou uma resposta inválida.";
    return fallback;
  }

  function linhaContribuicao(item) {
    const linha = document.createElement("div");
    linha.className = "contrib-row";
    const nome = document.createElement("span");
    nome.textContent = item.feature;
    const valor = document.createElement("b");
    const positivo = item.contribution >= 0;
    valor.className = positivo ? "contrib-pos" : "contrib-neg";
    valor.textContent = `${positivo ? "+" : ""}${item.contribution.toExponential(3)}`;
    linha.append(nome, valor);
    return linha;
  }

  explicar.addEventListener("click", async () => {
    const contexto = contextoAtual();
    if (!contexto) return;
    explicar.disabled = true;
    estado.textContent = "Calculando explicação...";
    saida.replaceChildren();
    try {
      const resposta = await fetch("explain", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(contexto.entrada),
      });
      if (!resposta.ok) throw new Error(String(resposta.status));
      const dados = await resposta.json();
      const ex = dados.explanation;
      estado.textContent = "Explicação pronta.";

      const card = document.createElement("section");
      card.className = "agente-card";
      const titulo = document.createElement("h4");
      titulo.textContent = `Contribuições — score ${dados.prediction.model_score.toFixed(3)}`;
      card.append(titulo);
      ex.contributions.slice(0, 8).forEach(item => card.append(linhaContribuicao(item)));
      if (ex.out_of_training_range.length) {
        const aviso = document.createElement("p");
        aviso.className = "faixa";
        aviso.textContent = `Fora da faixa de treino: ${ex.out_of_training_range.join(", ")}`;
        card.append(aviso);
      }
      saida.append(card);
    } catch (erro) {
      estado.textContent = mensagemErro(Number(erro.message), "Não foi possível explicar a previsão.");
    } finally {
      explicar.disabled = false;
    }
  });

  relatorio.addEventListener("click", async () => {
    const contexto = contextoAtual();
    if (!contexto) return;
    relatorio.disabled = true;
    estado.textContent = "Gerando relatório técnico...";
    saida.replaceChildren();
    try {
      const resposta = await fetch("reports/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(contexto.entrada),
      });
      if (!resposta.ok) throw new Error(String(resposta.status));
      const dados = await resposta.json();
      const report = dados.report;
      estado.textContent = "Relatório gerado.";

      const painel = document.createElement("section");
      painel.className = "agente-card agente-report";
      function bloco(titulo, texto) {
        const h = document.createElement("h4");
        h.textContent = titulo;
        const p = document.createElement("p");
        p.textContent = texto;
        painel.append(h, p);
      }
      function lista(titulo, itens) {
        const h = document.createElement("h4");
        h.textContent = titulo;
        const ul = document.createElement("ul");
        itens.forEach(texto => {
          const li = document.createElement("li");
          li.textContent = texto;
          ul.append(li);
        });
        painel.append(h, ul);
      }
      bloco("Resumo", report.summary);
      lista("Principais fatores", report.main_factors);
      lista("Limitações", report.limitations);
      lista("Ações recomendadas", report.recommended_actions);
      bloco("Conclusão", report.conclusion);
      saida.append(painel);
    } catch (erro) {
      estado.textContent = mensagemErro(Number(erro.message), "Não foi possível gerar o relatório.");
    } finally {
      relatorio.disabled = false;
    }
  });
};
