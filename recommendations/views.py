from django.shortcuts import render
from movie.models import Movie
from openai import OpenAI
from os import path
import os
from dotenv import load_dotenv
import numpy as np

def recommendations(request):
    best_movie = None
    similarity = None
    prompt = None
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        if prompt:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../openAI.env'))
            client = OpenAI(api_key=os.environ.get('openai_apikey'))
            response = client.embeddings.create(
                input=[prompt],
                model="text-embedding-3-small"
            )
            prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)
            max_similarity = -1
            for movie in Movie.objects.all():
                if movie.emb:
                    movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                    sim = np.dot(prompt_emb, movie_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(movie_emb))
                    if sim > max_similarity:
                        max_similarity = sim
                        best_movie = movie
                        similarity = sim
    return render(request, 'recommendations.html', {
        'best_movie': best_movie,
        'similarity': similarity,
        'prompt': prompt
    })
