---
layout: post
title:  "KNN'i Sevmiyoruz - 2"
date:   2022-01-13 12:33:07 -0800
categories: yazi
---

# Tek yol ortalama almak mı?

![](https://miro.medium.com/max/866/1*aGjYMi4j0XX4OQlaf6NM4Q.png)

Evet sevgili dostlar, yalan değil, rüyamda KNN gördüğüm geceler oluyor.

Bir önceki yazımızda KNN’de farklı mesafe ölçümlerinin yarattığı etkilere değinmiştik. İlk yazının detaylarına [buradan](https://medium.com/machine-learning-t%C3%BCrkiye/knni-sevmiyoruz-1-4cb5095c483e) erişebilirsiniz. Bu yazıda kısaca komşu sayısının etkileri ve komşular belirlendikten sonra kullanılabilecek tahmin metotları üzerinde duracağız.

#### Örnek Veri Seti

Örnek bir veri seti üzerinde komşu seçiminin ekstrem opsiyonları için eğitim ve test performanslarını inceleyelim.
Etiketleri oluşturan fonksiyonumuz $$Y = 3X^2 + N~(0, \sigma)$$ ve bağımsız değişkenimiz X de U[-8, 8] aralığından gelsin.

![](https://miro.medium.com/max/875/1*fRcHw-FogUaMRjKMSAgMjA.png)

#### Komşu Sayısı

Tıpkı mesafe ölçümü seçiminde olduğu gibi, komşu sayısı seçimi de alan bilgisinin etkin rol oynadığı ve farklı veri/örnekleme tiplerine göre farklılık arz edebilecek bir hiperparametredir.
Uç noktalar için, yani K=1 ve K=N senaryoları için bir KNN regresyon modelinin tahminlerini inceleyerek başlayabiliriz.

![](https://miro.medium.com/max/875/1*ewLKZePV-7WQVLgQLpFjLQ.png)

Yukarıda verilen grafiklere göre KNN modelinin eğitim seti üzerindeki performansı hakkında çıkarımlar yapmak gerekirse, K=1 durumu için KNN modelinin eğitim seti üzerinde tüm noktaların değerini tam olarak doğru tahmin ederek 0 hataya, ya da diğer bir deyişle azami regresyon başarısına ulaştığını görüyoruz. Diğer yandan, K=N senaryosunda ise modelin tüm tahminlerinin eğitim seti ortalamasına eşit olduğu gözleniyor. Standart bir regresyon başarısı ölçümü olarak R² skoru kullanıldığında K=N durumunda modelimizin skorunun 0 olduğunu görüyoruz — yani tüm gözlemlerin ortalamasını cevap olarak veriyoruz.

![](https://miro.medium.com/max/875/1*_YtcZqgYHeUl2zBFLNVJng.png)

Amacımız eğitim verisini ezberlemekten ziyade bir genelleme bulmak olduğu için, genellemeyi gerçekleştirebilecek doğru sayıdaki komşunun seçimi performans açısından kritik.

#### Komşuları bulduktan sonra?

Komşuları tespit ettikten sonra tek yöntem y değerlerinin ortalamasını almak ya da tasniflendirme için konuşursak oy çokluğuna başvurmak mıdır?
Yalnızca sorgu noktasına en yakın k komşu dikkate alınarak öğrenilen bir lineer fonksiyon tahminde kullanılabilir mi?

Bu noktada karşımıza 3 alternatif çıkıyor:

**(1) Kare hatayı yalnızca en yakın k komşuyu göz önüne alarak minimize edecek bir fonksiyon öğrenmek;**

[{}]()
$$E_1(x_q) = \frac{1}{2} \sum_{x \in x_q'nun\ en\ yakın\ k\ komşusu}{(f(x) - \hat{f}(x))^2}$$

![Sorgu noktasına en yakın K komşu ile oluşturulan regresyon hattı. Komşuların regresyon hattındaki payı eşit.](https://miro.medium.com/max/875/1*ZY-sKUz4kri-2L1YPJ-Sqw.png)

[İbrahim Halil Kaplan](https://medium.com/u/dd99dfacf7df?source=post_page-----9b1631da4fcd-----------------------------------) yazısında bu alternatif yöntemin detaylarına dikkat çekerek ve uygulamalı bir şekilde güzel örnekler veriyor:

[Localized Regression (KNN with Local Regression) | by İbrahim Halil Kaplan | Machine Learning Turkiye | Jan, 2022 | Medium
](https://medium.com/machine-learning-t%C3%BCrkiye/localized-regression-knn-with-local-regression-7b4d302adb85)

![](https://miro.medium.com/max/1250/1*tVuIXnZClK30Qvca2MogZA.png)

**(2) Hatayı tüm veri seti üzerinde minimize ederken hataları sorgu noktası xq’ya olan mesafelerine göre ağırlıklandırmak;**

{}
$$E_2(x_q) = \frac{1}{2}\sum_{x\in D}{(f(x) - \hat{f}(x))^2 K(d(x_q, x))}$$

![](https://miro.medium.com/max/875/1*BatljCUNuvBATcIb00Bwog.png)

Yukarıdaki şekilde verilen bir sorgu noktası için çizilen regresyon hattına tüm eğitim seti mesafelerine göre katkıda bulunuyor. Bu uygulamada ağırlıklandırma olarak RBF kullandık:

{}
$$K(d(x_q, x)) = e^{-\gamma || x_q - x_i ||_2}$$

fakat azalan mesafe ile artan farklı alternatif benzerlik fonksiyonları kullanmak da mümkün.
Her ne kadar bu yöntem akla en yatkın olanı gibi gözükse de her sorgu noktası için tüm veri seti üzerinde bir öğrenici eğitmek oldukça maliyetli.

```python
class WeightedRegressor(neighbors.KNeighborsRegressor):
    def __init__(self, gamma=None):
        super().__init__()
        if gamma is None:
            gamma = 1.0
        self.gamma = gamma
def predict(self, X):
        y_preds = []
        k = self._fit_X.shape[0]
        dist, inds = self.kneighbors(X, n_neighbors=k)
        for i in range(X.shape[0]):
            model_ftp = linear_model.LinearRegression()
            weights = np.exp(-self.gamma * dist[i])
#             print(weights)
            model_ftp.fit(self._fit_X[inds[i]], self._y[inds[i]], sample_weight=weights)
            y_preds.append(model_ftp.predict(X[i:i+1]))
        return np.array(y_preds)
```

![](https://miro.medium.com/max/1250/1*F4zj0bb9D2P5NB4sH2IcUw.png)


**(3) Komşuların eş katkısı (1) ve ağırlıklandırma metodlarının (2) bir kombinasyonu ve genellemesi olarak;**

{}
$$E_3(x_q) = \frac{1}{2}\sum_{x \in x_q'nun\ en\ yakın\ k\ komşusu}{(f(x) - \hat{f}(x))^2 K(d(x_q, x))}$$

Bu yöntem, hem (2)’nci metoda çok yakın olması hem de lokal model eğitim maliyetinin eğitim seti büyüklüğünden bağımsız olması itibariyle ideal bir çözüm olabilir.
Bu uygulamada da, (2)’de olduğu gibi ağırlıklandırmayı RBF kullanarak yaptık.

![](https://miro.medium.com/max/875/1*2JoXD4Ks_kcBKE0S66pvMQ.png)

```python
class LocallyWeightedRegressor(neighbors.KNeighborsRegressor):
    def __init__(self, n_neighbors=2, gamma=None):
        super().__init__(n_neighbors=n_neighbors)
        if gamma is None:
            gamma = 1.0
        self.gamma = gamma
def predict(self, X):
        y_preds = []
        dist, inds = self.kneighbors(X)
        for i in range(X.shape[0]):
            model_ftp = linear_model.LinearRegression()
            weights = np.exp(-self.gamma * dist[i])
            model_ftp.fit(self._fit_X[inds[i]], self._y[inds[i]], sample_weight=weights)
            y_preds.append(model_ftp.predict(X[i:i+1]))
        return np.array(y_preds)
```
![](https://miro.medium.com/max/1250/1*Qk2laUNHa5BC4a1sdPxrZg.png)

#### Sonuç — Farklı lokal yöntemler

Yukarıdaki örneklerde de görüldüğü üzere, komşu sayısını ve mesafe türünü seçtikten sonra tek opsiyon komşuların ortalamasını vermekten ibaret değil. İşlem kolaylığı açısından örneklerde lineer regresyonu kullandık fakat yöntem herhangi bir regresyon modeline — ister ağırlıklı olsun, ister eş ağırlıklarla — kolaylıkla genellenebilir. Lokalize karar ağacı mı; neden olmasın?
Lokalize lineer regresyon için detaylı örneklere [buradan](https://github.com/ibrakaplan/Local-Regression-KNN-with-Local-Regression) erişebilirsiniz.

Teşekkürler :)
