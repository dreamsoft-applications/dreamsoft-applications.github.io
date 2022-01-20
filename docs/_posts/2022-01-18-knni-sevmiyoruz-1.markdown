---
layout: post
title:  "KNN'i sevmiyoruz - 1"
date:   2021-07-27 21:14:37 -0800
categories: yazi
---

*Bence KNN güzel değil. LSTM daha iyi*

Evet, Hyundai i20'yi de sevmiyoruz, Volvo S80 daha iyi. Fakat öncelikle, KNN'in de dahil olduğu bellek bazlı öğrenicilere şöyle bir göz atmakta fayda var.

## Örnek bazlı öğrenme algoritmaları
Daha önce parametrik metotlarda öğrenme algoritmasına etiketlenmiş girdi — çıktı çiftlerinin girip girdiyi çıktıya haritalayacak parametrik bir fonksiyon (örneğin lineer regresyon için ağırlıklar ve sabit terim) ya da bölümlendirme biçimi (ağaç bazlı algoritmalar için) elde etmiştik.

![Kaynak filan](https://miro.medium.com/max/755/1*DyINblylU9RHBfNFU1Cc9Q.png)

Bellek bazlı öğrenme metotlarında durum bundan biraz daha farklı. Bu kez bize verilen girdi çıktı çiftlerini olduğu gibi bir veri tabanına atacağız ve öğrenme algoritmamız bize bir arama fonksiyonu verecek. Tasniflendirilmek ya da değeri tahmin edilmek istenen yeni örneklemler verildiğinde ise veri tabanımıza dönüp arama fonksiyonumuzu kullanarak sorunun cevabını bulacağız.

![](https://miro.medium.com/max/875/1*Vw89z5-kKLJkcGToZbS0vQ.png)

Bu yaklaşımın artıları ve eksileri neler olabilir?
- Eğitim sırasında sağlanan veriler saklanmaktadır. Dolayısıyla yeni veri elde edildiğinde baştan eğitim söz konusu değildir; yeni veri, veri tabanına eklenir ve aramaya dahil olur.
- Öğrenme aşaması hızlıdır (öğrenecek bir şey olmadığından). Bu açıdan örnek bazlı öğreniciler “tembel öğrenici” olarak da adlandırılmaktadır.
- Basit, açıklanabilir kararlar

Özellikle bu noktada basitlik/anlaşılabilirlik hususuna dikkat çekme gereği görüyoruz. Genelde modellerinizin tüketicileri alanında uzman fakat makine öğrenmesine yabancı kişilerden oluşacaktır. Bu noktada tüketiciler modeli kullanmaya ikna olmak için modelin çalışma prensibi ve verilen kararların açıklamalarını sorgulayacaktır. Yalnızca girdileri ve çıktıları belli olup arasındaki süreç anlaşılamayan kara kutu tipinde bir model, kullanıcıların kullanmaktan kaçınmasına sebep olabilir.

![](https://miro.medium.com/max/733/0*bv-EoIzPEZ1FEHSc)


Eksileri:
- Genelleme yeteneği düşük, veriyi ezberleme olayı meydana gelmekte (bu kez mecazi değil gerçek)
- Tamamen enterpolasyona yönelik: Gözlemlenen dağılımın dışından gelen bir veri noktası için parametrik fonksiyonlarda olduğu gibi çıkarım yapmak mümkün değil.
- Bellek üzerinde ağır yük (tüm eğitim verisinin tutulma zorunluluğu). Aksi bir örnek olarak, lineer regresyonda katsayılar öğrenildikten sonra eğitim verisine ihtiyaç duyulmuyordu.
- Uzayabilen tahmin süreleri. Eğitim süresi veri miktarından bağımsız olsa da tahmin süresi, veri miktarı ve boyutu ile artmaktadır.


Sık kullanılan örnek bazlı öğrenicilerden KNN algoritmasını inceleyelim.

## KNN - K-en yakın komşu

KNN algoritmasının işleyişini özetlemek gerekirse, verilen bir sorgu noktası ve bir eğitim seti için, belirli bir mesafe ölçümünü kullanarak eğitim setinden en sorgu noktasına en yakın K komşuyu belirler. İkinci aşamada, sorgu noktası için yapılacak tahmin,

- regresyon problemleri için en yakın komşuların hedef değerlerinin ortalaması ya da,
- tasniflendirme problemleri için basit bir oylama ile en yakın komşulardan çoğunluk hangi sınıfa ait ise sorgu noktası için de bu sınıf tahmin edilebilir.

Verilen karar oy çokluğu ya da hedef değişkenin ortalaması olabileceği gibi, komşuların karar üzerindeki etkilerini sorgu noktasına olan mesafelerine göre ağırlıklandırmak da mümkündür.

Algoritmayı kısaca formül haline getirmek gerekirse:
- Eğitim verisi: D={xi, yi}
- Mesafe ölçüsü: d(q, x)
- Komşu sayısı: K
- Sorgu noktası: q

Bu formüle göre verilen q noktası için en yakın komşular {En küçük K i: d(q, xi)} bulunduktan sonra problemin türüne göre algoritmanın bir sonraki bileşeni tahmin mekanizması olacaktır.

- Tahmin mekanizması: f({<Xi, Yi>: d(q, Xi) en küçük olan K})

Burada mesafe ölçümü d, komşu sayısı K ve komşuların karara etkisi (düzgün ya da mesafe ağırlıklı), alan bilgisinin devreye girdiği parametrelerdir. Farklı iş alanları farklı tür mesafe ölçümlerinin kullanılmasını gerektirebileceği gibi, doğru komşu sayısı seçimi de uygulama alanının gereksinimlerine ve verinin mahiyetine göre değişiklik gösterecektir.

Aşağıda 1-NN (1 en yakın komşu) regresyon algoritması için farklı mesafe ölçümleri ile oluşturulmuş tahmin şemaları görülmektedir.

![(Solda) Öklid mesafelerine göre oluşturulmuş en yakın 1 komşu algoritması için Voronoy diyagramı. (Sağda) Aynı Voronoy diyagramının büyük ölçekte görünümü.](https://miro.medium.com/max/875/1*kwUMCx-hqku-eufvuWw-LQ.png)

![p=1 Minkowski mesafelerine (Manhattan mesafesi) göre oluşturulmuş Voronoy diyagramlarının büyük ve küçük ölçekte görünümü](https://miro.medium.com/max/875/1*dXtZCpvvBCiADzR7gvvoeg.png)

![P=1000 Minkowski mesafelerine göre oluşturulmuş Voronoy diyagramlarının büyük ve küçük ölçekte görünümü.](https://miro.medium.com/max/875/1*aF0lcacgA-PAj6G1jY0diw.png)

![Kosinüs mesafelerine göre oluşturulmuş Voronoy diyagramının büyük ve küçük ölçeklerde görünümleri.
](https://miro.medium.com/max/875/1*FsBSnGbQPMXVpErre0iEZQ.png)

## Önyargılar, önyargılar…
Pek çok modelde olduğu gibi, KNN modelini kullanmaya karar verdiğinizde de algoritma bir takım önyargıları tahminlerine yansıtacaktır. Bunlardan bazıları aşağıdaki gibi özetlenebilir:
- Yerellik: birbirine yakın noktalar benzerdir.

Algoritmanın isminden de anlaşılacağı üzere, KNN kullandığınızda hedef değişkeni oluşturan fonksiyonun yakın noktalar için benzer değerler verdiği hipotezleri tercih ediyorsunuz.

- Tüm değişkenler aynı öneme sahiptir.

Dizayn matrisinizin tüm boyutları, ağırlıklandırılmaksızın, komşu aramasında mesafeye eşit derecede katkıda bulunur.

![](https://miro.medium.com/max/796/1*AFxE7y6dqxKQtL_uvNld7g.png)

Yukarıda gayrimenkul sayısı ile doğrusal ilişkisi net bir şekilde görülen Kart Harcama miktarı için, KNN (1 komşu) modelinin 6 gayrimenkule sahip müşteri için yapacağı tahmin 30.000 olacaktır (çünkü dün akşam çok tavuk yedi).

Mesafelerden ve değişkenlerin eş katkısından bahsetmişken, ikinci ve daha vahim bir örnek sunabiliriz. Bu örnek için ekmek fiyatının 100.000 TL olduğu 2000 senesine gidiyoruz.

![](https://miro.medium.com/max/225/1*10jY7IkSPk0YEFDmw-yljA.png)

Herhangi bir ölçekleme yapmaksızın, son satırdaki verinin kalan örnekler arasından en yakın komşusunu bulmaya çalıştığımızda karşımıza çıkan manzara aşağıdaki gibi olacaktır.

![](https://miro.medium.com/max/875/1*MDOU2E1Izn0pH3ptwl6pKg.png)

Ekmeğe ödenen para miktarının anlam ifade edip etmemesi bir yana, gayrimenkul sayısına kıyasla çok büyük bir ölçekte değişiklik gösteren bu değer, mesafe ölçümünde daha küçük ölçekte değişen fakat dağılımın uç noktasından gelmiş olan gayrimenkul sayısını gölgeleyerek karar üzerinde en büyük etkiyi yapıyor.


Yukarıdaki örnekte görüldüğü üzere, mesafe ölçümü ya da benzerliğin söz konusu olduğu algoritmalarda değişkenlerin standart bir ölçeğe getirilmesi mantıklı görünüyor. Ölçekleme yöntemleri konusunda detaylı bilgi için; [Feature Scaling with Scikit-Learn for Data Science \| by Hasan Ersan YAĞCI \| Medium](https://hersanyagci.medium.com/feature-scaling-with-scikit-learn-for-data-science-8c4cbcf2daff)

Özet olarak, bu yazımızda örnek bazlı öğrenicilerden KNN’yi kısaca tanıyıp algoritmanın temel bileşenleri ve yine temel seviyede işleyişi üzerinde durduk. Ek olarak, mesafe bileşeninin algoritmanın kararlarına etkileri ve KNN’in önyargılarına değindik. Serinin bir sonraki yazısında, algoritmanın bir diğer bileşeni olan komşu sayısı K’nın genelleme kabiliyeti üzerindeki etkisinden ve komşuların tespitinden sonra uygulanabilecek alternatif tahmin metotlarından bahsedeceğiz.

## Referanslar
Isbell, C., Litmann, M., “Instance Based Learning” https://classroom.udacity.com/courses/ud262#
