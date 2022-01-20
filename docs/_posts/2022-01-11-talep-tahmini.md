---
layout: post
title:  "Bi-LSTM ile Talep Tahmini"
date:   2022-01-11 12:33:07 -0800
categories: yazi
---



***Editörün notu:*** *Elinizdeki bir mala olan talebi isabetli bir şekilde bilebilmek nelere kumanda ederdi? Ya da hatalı bir tahmin nelere sebep olabilirdi? Gerçek değerin çok üzerinde bir tahmin ya malın elde kalması sebebiyle zarara sebep olur, ya da satıcıyı aşırı indirim vermeye zorlayarak hem kârdan feragate hem de marka değerinin düşmesi ile sonuçlanabilir. İşte perakendeciliğin tam da kalbinde yer alan bu kritik süreç, talep tahmini meselesini [Ersan](https://hersanyagci.medium.com/)* ***Derin öğrenme*** *alanındaki son gelişmeler ışığında ele alıyor.*


***Özet:*** *Ekonominin temel kavramlarından olan arz talep dengesi her ticari kuruluşun en önemli maddesidir. Ticari firmaların sattıkları ürüne ya da verdikleri hizmete olan talebi önceden bilmeleri karı maksimum edebilmeleri için hayati önemdedir. Bu çalışmada, bir şehirdeki bisiklet kiralamasına olan talep çift yönlü uzun kısa süreli bellek ağı (bi-LSTM) yöntemi ile tahmin edilmiştir. Verimizde saatlik veri bulunmaktadır ve 2 gün ilerisi tahminlenmiştir.*

## 1. Giriş
Son zamanlarda yaşanan salgın ve salgının yarattığı küresel ekonomik tedirginlikler bir mala olan ihtiyacı önceden belirleyerek, buna karşı iktisadi çözümler alabilmek maksadıyla çokça önem kazanmıştır. Talep tahmini genel olarak zaman serisi tahminleri ile yapılmaktadır. Ancak son zamanlarda, tahminlemede modern derin öğrenme teknikleri kullanılmaya başlandı. Bunların başında özyinelemeli sinir ağları (RNN) gelmektedir.

Ancak RNN’in bazı problemlerde yaşadığı hafıza sorununu çözmek maksadıyla, LSTM ve bunun bir ileri versiyonu olan bi-LSTM tercih edilmeye başlanmıştır. Tabi problemin ve verinin türüne göre hangi yöntemle daha başarılı sonuçlar alınabileceği değişmektedir.

Bir çok çalışmada hibrit modeller kullanılmıştır. [1]’de bir şehre ait su ihtiyacı evrişimli sinir ağları (CNN) ve bi-LSTM hibrit modeli ile tahminlenmiştir. Çalışmada CNN, LSTM ve bi-LSTM hem tek tek deneylenmiş hem de hibrit olarak kullanılmışır.

Bu çalışmada [2] ise, LSTM modeli hem ARIMA modeli ile hem de geri yayılım derin öğrenme (BPNN) yöntemi ile kıyaslanarak, turizm büyüklüğü, ziyaretçi miktarı tahminlenmeye çalışılmıştır. LSTM’in daha iyi sonuç verdiği çalışmada kullanılan veri tahminlemenin çok farklı alanlarda kullanabileceğini göstermektedir.

LSTM çok geniş bir alanda kullanılabilen güçlü bir tahminleme modelidir. Bu çalışmada [3] ise, konuşma tanıma hibrit bir model olan derin bi-LSTM yöntemi ile yapılmıştır. LSTM kullanıldığı hibrit modellerde de iyi sonuç vermektedir.

Hem tüketim miktarının belirlenmesi hem de üretimin miktarını belirleyebilmek maksadıyla ihtiyaçların belirlenmesinde LSTM yönteminin hem yalnız hem de hibrit modeller içinde sıkça kullanıldığı görülmektedir [4].

Çalışmamızda ise bir şehre ait bisiklet kiralamasına ait talepler tahminlenmiştir. Bisiklet kiralamasına olan talepler, aynı yapıda olan bir veri ile bir şehrin su ve elektrik talebi veya bir ticari firmanın kendi ürününe olan ihtiyacı taleplemesi olarak düşünülüp kullanılabilir.

## 2. Veri
Çalışmamızda bir şehrin 2015-2017 yılları arasındaki saatlik bisiklet kullanım verisi kullanılmıştır [5]. Şehre ait bisiklet istasyonlarından; o saatte alınan bisiklet miktarları bağımlı değişkenimiz, tahminini yapacağımız değişkendir. Verimizde sıcaklık, nem, rüzgar hızı, hava durumu, haftasonu ve tatil olup olmama durumları, mevsim ile zaman bağımsız değişkenlerimizdir.

![](/images/erso-tt/erso-tt-1.png)

Verimizde eksik veri bulunmamaktadır. Uygulama öncesinde veriye keşifsel veri analizi uygulanmış, görselleştirmeler ile aykırı değerler ve veri içindeki paternler keşfedilmeye çalışılmıştır.

 Veriler farklı zaman dilimlerinde çeşitli açılardan gözden geçirilmiştir. Örnek olarak, bisiklet paylaşımlarının haftanın günlerine göre dağılımı görselleştirilerek, bu grafik ile insanların davranışlarının günlük olarak nasıl değiştiği gözlemlenmiştir. Aynı şekilde saatlik, aylık, mevsimsel analizler de yapılmıştır

 ![](/images/erso-tt/erso-tt-2.png)

 Çalışmamızın amacı sonraki dönemlere ait talep tahmininde bulunmaktır. Ancak verinin modelleme öncesindeki analizinin başarıya etkisi bilindiğinden detaylı analizler ve ön işlemler yapılmıştır.

 Verimiz eğitim ve test olarak bölündükten sonra, eğitim ve test verilerine ölçekleme (robust scaling) uygulanmıştır. Ölçeklemenin amacı veride mevcut outlierların etkisini kırmaktır. Daha sonrasında verimiz kullanacağımız yönteme uygun olarak üç boyutlu bir seriye dönüştürülmüştür.


## 3. Yöntem


 Çalışmada çift yönlü LSTM (Bidirectional LSTM) kullanılmıştır. LSTM’den önce özyinelemeli sinir ağından (RNN) bahsetmek daha uygun olacaktır. Çünkü LSTM bir RNN mimarisidir. RNN, metin, genom, el yazısı, konuşulan kelime, sayısal zaman serisi verileri gibi veri dizilerindeki kalıpları tanımak için tasarlanmış bir tür yapay sinir ağıdır.

 RNN, eğitim için geri yayılım algoritmasını kullanır. Dahili hafızaları nedeniyle, RNN'ler aldıkları girdiyle ilgili önemli şeyleri hatırlayabilir, bu da bir sonraki adımı tahmin etmede çok kesin olmalarını sağlar. Örnek vermek gerekirse, okuduğumuz bir cümleyi anlamamız cümlenin önceki kelimelerini de hatırlamamız sayesindedir. RNN yapısı da önceki bilgileri kullanarak ileriyi tahminlemede kullanılır. Her ürettiği çıkışı bir önceki adıma bağlayarak ilerletir.

 RNN bir tanh katmanı içerirken, LSTM 4 farklı katman içermektedir. LSTM, bilgileri depolayarak RNN’in kısa vadeli bellek problemini ortadan kaldırmaktadır

 ![](/images/erso-tt/erso-tt-3.png)

 Çalışmamızda, LSTM sıralı öğrenme modeli, sıralı verileri işleme ve geçmiş zaman adımlarının verilerini ezberleme yeteneği nedeniyle seçilmiştir. Burada model performansını daha da iyileştiren çift yönlü LSTM kullanılmıştır.

Çift yönlü LSTM'ler giriş dizisinde bir LSTM yerine iki tane eğitir. İlki, olduğu gibi giriş dizisinde ve ikincisi, giriş dizisinin ters çevrilmiş bir kopyasında. Bu, ağa ek bağlam sağlayabilir ve problem üzerinde daha hızlı ve hatta daha eksiksiz öğrenme ile sonuçlanabilir. Figure 4, Çift Yönlü LSTM'nin mekanizmasını göstermektedir.

![](/images/erso-tt/erso-tt-4.png)

## 4. Uygulama
Verimiz modellemeye uygun hale getirildikten sonra Figure 5’te gösterilen yapay sinir ağı uygulanmıştır. Parametre ayarlaması sonrasında optimizer olarak adam optimizer, kayıp fonksiyon olarak mse (mean squared error) seçilmiştir. 0.2 oranında dropout eklenmiş, batch size olarak 128 ve epoch sayısı 200 uygulanmıştır.

![](/images/erso-tt/erso-tt-5.png)

Model sonucunda loss ve validation loss grafikleri çizdirilmiş,  Figure 6’da sunulmuştur.

![](/images/erso-tt/erso-tt-6.png)

Tahmin yaparken, gerçek bir hata puanı elde etmek için yapılan ölçeklendirmeyi ters dönüştürme işlemi yapılmıştır. Son olarak, modelin tahminlerinin test verilerinin gerçek değerleriyle ne kadar iyi eşleştiği görselleştirilmiş ve ayrıca hata puanları hesaplanmıştır. R skor 0.96, MAE (Mean Absolute Error) ise 100 çıkmıştır. Figure 7’de birinci grafikte yeşil geçmiş verileri, mavi tahminlenen zamanın gerçek değerlerini, kırmızı ise tahminleri göstermektedir. İkinci grafik ise test ve gerçek verileri daha ayrıntılı vermektedir.

![](/images/erso-tt/erso-tt-7.png)


|Method	|Accuracy Score|
|---	| ---|
|Attention GANs	|97.69
|VGG-16 CapsNet (CNN)|	95.21
|MARTA GANs	|94.68
|GoogleNet (CNN)|	94.31
|SCDAE (Stacked AE)|	93.7
|SGUFL (AE)|	82.72

## 5. Sonuç ve Tartışma
Bu çalışmada son zamanlarda bir çok alanda kullanılan çift yönlü LSTM ile talep tahmini yapmak üzere bir deney yapılmıştır. Deney sonucunda tahminlerin başarısının yüksek olduğu gözükmektedir. Burada kullanılan çift yönlü LSTM, farklı hibrit modeller içerisinde kullanılarak çalışma ileriye taşınabilir. Bulunan en uygun yöntem ise hayati öneme haiz, insan ihtiyaçlarını karşılamak üzere olan bir veri üzerinde  kullanılabilir.

## Referanslar
- [1] 	P. Hu, J. Tong, J. Wang, Y. Yang and L. d. Oliveira Turci, "A hybrid model based on CNN and Bi-LSTM for urban water demand prediction," 2019 IEEE Congress on Evolutionary Computation (CEC), 2019, pp. 1088-1094, doi: 10.1109/CEC.2019.8790060.
- [2] 	YiFei Li, Han Cao, Prediction for Tourism Flow based on LSTM Neural Network, Procedia Computer Science, Volume 129, 2018, Pages 277-283, ISSN 1877-0509.
- [3] 	Graves, A., Jaitly, N., Mohamed, A.: Hybrid Speech Recognition with Deep Bidirectional LSTM[J], Automatic Speech Recognition and Understanding (ASRU), (2013).
- [4] 	Jiujian, Wang & Wen, Guilin & Yang, Shaopu & Liu, Yongqiang. (2018). Remaining Useful Life Estimation in Prognostics Using Deep Bidirectional LSTM Neural Network. 1037-1042. 10.1109/PHM-Chongqing.2018.00184.
- [5]    [https://github.com/hersany/DataScience/tree/master/MSKU/Demand_Prediction](https://github.com/hersany/DataScience/tree/master/MSKU/Demand_Prediction)
