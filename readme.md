# 🚀 Fine-Tuning de Modelo com 50k Dados

Este repositório documenta o processo completo de **fine-tuning de um modelo de linguagem** utilizando **50.000 amostras de dados**.  
O pipeline inclui **preparação do dataset**, **treinamento no Google Colab** e **comparação do modelo fine-tuned com o modelo base**.  

---

## 📂 Estrutura do Repositório

- **`prepare_data_22_09.ipynb`**  
  Responsável por preparar o dataset de entrada.  
  - Limpeza e padronização  
  - Divisão em treino e validação  
  - Exportação no formato compatível com Hugging Face Datasets  

- **`fine_tuning_50k_colab_22_09.ipynb`**  
  Notebook que executa o fine-tuning do modelo.  
  - Configuração do ambiente (Google Colab + GPU)  
  - Carregamento do modelo base (Hugging Face Transformers)  
  - Treinamento com 50k amostras  
  - Salvamento do modelo fine-tuned  

- **`compare_models_final_22_09.ipynb`**  
  Avaliação e análise de desempenho.  
  - Comparação entre modelo base e modelo fine-tuned  
  - Métricas como loss, acurácia e perplexidade  
  - Gráficos de evolução do treinamento  
  - Conclusões sobre os ganhos obtidos  

---

## ⚙️ Requisitos

- Python 3.8+  
- Google Colab ou ambiente com GPU CUDA  
- Bibliotecas principais:  
  - `transformers` (Hugging Face)  
  - `datasets`  
  - `torch`  
  - `pandas`, `numpy`, `matplotlib`  

Instale os pacotes necessários com:  

```bash
pip install torch transformers datasets pandas matplotlib
```

---

## 🚀 Como Reproduzir

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repo.git
   cd seu-repo
   ```

2. Execute os notebooks na seguinte ordem:
   1. `prepare_data_22_09.ipynb` → Gera o dataset final.  
   2. `fine_tuning_50k_colab_22_09.ipynb` → Treina o modelo.  
   3. `compare_models_final_22_09.ipynb` → Analisa os resultados.  

3. (Opcional) Publique o modelo no [Hugging Face Hub](https://huggingface.co/).  

---

## 📊 Resultados

O modelo treinado com **50.000 amostras** apresentou:  
- Redução significativa do **loss** em relação ao modelo base.  
- Melhoria nas métricas de avaliação (ex.: acurácia e perplexidade).  
- Melhor desempenho em inferências de teste qualitativas.  

👉 Os gráficos e métricas detalhadas estão no notebook `compare_models_final_22_09.ipynb`.

---

## 🖥️ Exemplo de Inferência

Após o fine-tuning, o modelo pode ser utilizado para inferência da seguinte forma:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "seu-usuario/seu-modelo-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

prompt = "Explique de forma simples o que é aprendizado de máquina."

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📌 Próximos Passos

- [ ] Publicar modelo no Hugging Face Hub  
- [ ] Testar em **benchmark externo** para validação mais robusta  
- [ ] Ampliar dataset (>100k amostras)  
- [ ] Experimentar técnicas de **LoRA** e **PEFT** para fine-tuning mais eficiente  

---

## 📜 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para mais detalhes.