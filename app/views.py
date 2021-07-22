from django.shortcuts import render
import os
from django.http import HttpResponse

def homepage(request):
	return render(request,'ML_homepage.html')

def graph1(request):
	os.system(r"python F:\project\titanicML\app\graph1.py")
	return render(request,"ML_homepage.html")

def graph2(request):
	os.system(r"python F:\project\titanicML\app\graph2.py")
	return render(request,"ML_homepage.html")

def graph3(request):
	os.system(r"python F:\project\titanicML\app\graph3.py")
	return render(request,"ML_homepage.html")

def histo(request):
	os.system(r"python F:\project\titanicML\app\histo.py")
	return render(request,"ML_homepage.html")

def predict(request):
	os.system(r"python F:\project\titanicML\app\predict.py")
	return render(request,"ML_homepage.html")

def genCSV(request):
	os.system(r"python F:\project\titanicML\app\genCSV.py")
	return render(request,"ML_homepage.html")