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
  raiz.innerHTML = '<p class="note">O agente de IA ainda não foi conectado.</p>';
};
