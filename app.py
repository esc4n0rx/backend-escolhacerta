from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

TMDB_API_TOKEN = os.getenv('TMDB_API_TOKEN')

def sim_nao(value):
    return 'Sim' if value else 'Não'

# Ajuste do prompt do sistema
system = """
Você é um recomendador de filmes e séries. Seu trabalho é entender as preferências do usuário e recomendar conteúdos adequados com base nas respostas fornecidas.
Por favor, forneça exatamente o número de recomendações solicitadas, nem mais nem menos. Liste as recomendações numeradas.
"""

human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
# Aumentando a temperatura para gerar respostas mais variadas
chat = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")
chain = prompt | chat

def get_popular_movies():
    url = "https://api.themoviedb.org/3/movie/popular?language=pt-BR"
    headers = {
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    try:
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            data = response.json()
            popular_movies = [movie["title"] for movie in data.get("results", [])]
            return popular_movies if popular_movies else None
        else:
            return None
    except Exception as e:
        print(f"Erro ao buscar filmes populares: {e}")
        return None

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('user_input', {})
    
    novidade = user_input.get('novidade', 0)
    prefere_filmes = user_input.get('prefere_filmes', False)
    quantidade = min(int(user_input.get('quantidade', 1)), 5)
    
    if novidade == 1 and prefere_filmes:
        popular_movies = get_popular_movies()
        if popular_movies:
            selected_movies = popular_movies[:quantidade]
            response_text = f"Com base no que você está procurando, eu recomendaria assistir: {', '.join(selected_movies)}. Esses são alguns dos filmes mais populares no momento!"
            return jsonify({"recommendation": response_text})
        else:
            return jsonify({"recommendation": "Não foi possível buscar filmes populares no momento."})
    
    # Coletando todos os parâmetros do usuário
    estado_emocional = user_input.get('estado_emocional', '')
    tempo_disponivel = user_input.get('tempo_disponivel', '')
    preferencias_genero_list = user_input.get('preferencias_genero', [])
    preferencias_genero = ", ".join(preferencias_genero_list)
    intensidade = user_input.get('intensidade', '')
    
    # Montando a consulta do usuário
    user_query = f"""
    Estado emocional: {estado_emocional}
    Tempo disponível: {tempo_disponivel} minutos
    Preferências de gênero: {preferencias_genero}
    Intensidade desejada: {intensidade}
    Prefere filmes: {sim_nao(prefere_filmes)}
    Quer novidades: {sim_nao(novidade == 1)}
    Quantidade de recomendações desejadas: {quantidade}
    """
    
    try:
        # Gerando a resposta do modelo
        response = chain.invoke({"text": user_query})
        response_text = response.content.strip()
    except Exception as e:
        print(f"Erro ao gerar resposta do modelo: {e}")
        response_text = "Desculpe, ocorreu um erro ao gerar a recomendação."
        return jsonify({"recommendation": response_text})
    
    # Processando a resposta do modelo
    recommendations = []
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            recommendation = line.split('.', 1)[-1].strip()
            recommendations.append(recommendation)
        elif line:
            recommendations.append(line)
    
    # Limitando ao número de recomendações desejadas
    recommendations = recommendations[:quantidade]
    
    if not recommendations:
        response_text = "Parece que não consegui encontrar sugestões específicas, mas você pode tentar buscar por clássicos como 'Indiana Jones' ou 'Star Wars'!"
    else:
        response_text = "\n\n".join(recommendations)
    
    return jsonify({"recommendation": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
