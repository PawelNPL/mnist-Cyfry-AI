import numpy
import tensorflow
import matplotlib
import keras
import os
import matplotlib.pyplot as plt
from keras.models import Sequential , load_model
from keras.layers import Dense, Flatten
from keras.datasets import mnist  #keras ma wbudowany dataset mnist

MODEL_PATH = 'model_cyfry.h5'

(X_train, y_train), (X_test, y_test) = mnist.load_data()   #ładujemy dane


X_train = X_train / 255.0
X_test = X_test / 255.0     #normalizacja

#zapisywanie wytrenowanego modelu
if os.path.exists(MODEL_PATH):
    print("model znaleziony, wczytywanie......")
    model = load_model(MODEL_PATH)
else:    
    print("model nie znaleziony, trenowanie nowego modelu")
    
    model = Sequential([                    #budowa modelu
        Flatten(input_shape=(28,28)),       #zmienia obrazek na wektor
        Dense(128, activation="relu"),      #relu zostawia wartość dodatnią, a wartość ujemną zmienia na 0
        Dense(10, activation="softmax")     #softmax jest używana na ostatniej warstwie, przekształca wektor liczb na rozkład prawdopodobieństwa
        
        ])
    
        #kompilacja
    model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])  
        #optymalizator adam, funkcja straty mierząca różnicę od oczekiwanych wyników, metryka sprawdza poprawność. 


        #trenowanie
    model.fit(X_train , y_train , epochs=10) 
        # epochs(epoki) - ilość przejść modelu przez dane treningowe
        #batch_size - rozmiar paczki uczenia domyślnie 32
  
    
    model.save(MODEL_PATH)
    print("Model wytrenoway i zapisany w pliku")




#sprawdzenie na danych testowych / ewaluacja
test_loss , test_accuracy = model.evaluate(X_test , y_test)
print(f'dokładność na danych testowych: {test_accuracy} \n test loss: {test_loss}')

#wizualizacje i predykcje
predictions = model.predict(X_test)


ilosc_przykladow = 5
for i in range(ilosc_przykladow):
    #Wyświetlanie obrazków
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    
    #Pobieranie etykiet
    predicted_label = numpy.argmax(predictions[i])
    true_label = y_test[i]
    
    #kolor tytułu
    if predicted_label == true_label:
        title_color = "green"
    else:
        title_color = "red"
        
    #dodawanie tytułu i wyświetlanie
    plt.title(f"Przewidywana: {predicted_label}, Prawdziwa: {true_label}", color=title_color)
    plt.show()
    
    
    
    
    
    
    
    #
    #  Dodane ekstra: opcja zapisania wytrenowanego modelu do pliku by nie musiał się cały czas trenować
    #