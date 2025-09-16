To repozytorium zawiera prosty model sieci neuronowej, zbudowany w TensorFlow/Keras, do klasyfikacji ręcznie pisanych cyfr ze zbioru danych MNIST.

Opis projektu

Celem projektu jest zbudowanie i wytrenowanie prostego klasyfikatora, który potrafi rozpoznać, jaka cyfra (od 0 do 9) znajduje się na obrazku. Zastosowany model składa się z następujących warstw:

Warstwa wejściowa (Flatten): Przekształca obrazek o wymiarach 28x28 pikseli w jednowymiarowy wektor.

Warstwa ukryta (Dense): 128 neuronów z funkcją aktywacji ReLU.

Warstwa wyjściowa (Dense): 10 neuronów (po jednym dla każdej cyfry) z funkcją aktywacji Softmax, która zwraca rozkład prawdopodobieństwa.**
