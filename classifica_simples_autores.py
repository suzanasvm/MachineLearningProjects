#!-*- coding: utf8 -*-
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Lendo dados
classificacoes = pd.read_csv('dataset.csv')
#apenas as frases
textosPuros = classificacoes['frase']
#quebrar os textos e transforma tudo em minusculo
textosMinusculo = textosPuros.str.lower().str.split(' ')

#Cria um dicionario que ignora as palavras repetidas
dicionario = set()
for lista in textosMinusculo:
	dicionario.update(lista)
#Exibe todo o dicionario	
print "Dicionario: "
print dicionario

#Atribui cada palavra a uma posicao no dicionario
totalDePalavras = len(dicionario)
tuplas = zip(dicionario, xrange(totalDePalavras))
tradutor = {palavra:indice for palavra,indice in tuplas}

#Mostra a quantidade total de palavras
print "Total de palavras: "
print totalDePalavras


def vetorizar_texto(texto, tradutor):
	vetor = [0] * len(tradutor)

	for palavra in texto:
		if palavra in tradutor:
			posicao = tradutor[palavra]
			vetor[posicao] += 1

	return vetor

#Vincula os textos quebrados a posicao no vetor
vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosMinusculo]
marcas = classificacoes['classificacao']

#Exibe todas as frases do dataset
print "Todas as frases do dataset: "
print textosMinusculo

#Define o conjunto de dados X
X = np.array(vetoresDeTexto)
#Define o conjunto de dados Y (labels)
Y = np.array(marcas.tolist())

#Define porcentagem do treino
porcentagem_de_treino = 0.5

#Separa o tamanho do treino a partir da porcentagem
tamanho_do_treino = int(porcentagem_de_treino * len(Y))
#O restante fica para a validacao
tamanho_de_validacao = (len(Y) - tamanho_do_treino)

print "Frases disponiveis: "
print len(Y)
print "Frases para treino: "
print tamanho_do_treino
print "Frase para validacao: "
print tamanho_de_validacao


#Separa os dados de treino
treino_dados = X[0:tamanho_do_treino]
#Separa as marcacoes de treino
treino_marcacoes = Y[0:tamanho_do_treino]
#Separa os dados de validacao
validacao_dados = X[tamanho_do_treino:]
#Separa as marcacoes de validacao
validacao_marcacoes = Y[tamanho_do_treino:]

print "Textos usados na validacao: "
print textosMinusculo[tamanho_do_treino:]
print "Validacao Marcacoes: "
print validacao_marcacoes

clf = GaussianNB()
clf.fit(treino_dados, treino_marcacoes)
clf.predict(validacao_dados)

accuracy = clf.score(validacao_dados, validacao_marcacoes)

print "Indice de acerto do algoritmo: "
print accuracy * 100