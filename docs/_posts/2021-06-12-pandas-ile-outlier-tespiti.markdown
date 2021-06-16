---
layout: post
title:  "Pandas ile Outlier Tespiti ve Mücadele Yöntemleri"
date:   2021-06-12 21:14:37 -0800
categories: yazi
---
Sayın [Atçılı](https://abdullahatcili.medium.com/)'nın [Machine Learning Turkiye](https://medium.com/machine-learning-t%C3%BCrkiye)'de 6 Haziran 2021 tarihinde yayınlanmış olan yazısıdır.

Data analizi oldukça uzun bir işlemdir. Bunun için yapılması gereken bir dizi adımlar mevcuttur. Öncelikle, datanın ne olduğunun tespit edilmesi önem arz etmektedir. Veri setindeki her bir kolonun (feature) neler olduğunu bilmek ve anlamak gerekmektedir. Bu adımdan sonra ise boş değerlikli (NaN — not a number-values) sütun tespiti yapılmalıdır. NaN değerler, veri setinden silinebilir veya farklı değerlerle doldurulabilir (Örneğin; mean, median gibi). Ya da NaN değerleri doldurmak için fonksiyon yazılıp detaylı olarak doldurulabilir. Veya çok fazla NaN değer içeren sütunların düşürülmesi dahi düşünülebilir.

Yukarıda belirtilen teknikler veriden veriye değişiklik göstermektedir. Ama ortada bir gerçek var ki bu da outlier değerlerden kurtulmamız gerektiğidir. Outlier değerler tespit edilip veri setinden soyutlanmalıdır. Her bir data 1.5 IQR içerisinde olup olmama durumuna göre farklı tiplerde outlierlar içerebilir. Bazı durumlarda da bahse konu outlierlar zararlı olmaz ve bunlara müdahale edilmek durumunda kalınmayabilir. Ancak modelimizde veya analizimizde daha başarılı ve sağlıklı sonuçlar istiyorsak, outlierları düzenlemek durumundayız. Bununla ilgili üç teknikten bahsetmek mümkün olacaktır.

1. Outlier değerlerin düşürülmesi
2. Winsorize yöntemi
3. Log Transformasyonu

![pandas](https://miro.medium.com/max/601/1*9fFrwMNbHhr3o28kEaey_w.png)


Şimdi bu methodlarla ilgili kodlara bakalım. Burada Seaborn kütüphanesine ait “diamonds” veri seti kullanılmıştır.

![gorsel 1](https://miro.medium.com/max/875/1*IhAKeVNPlkn4eJt6Mjbs-g.png)

Şimdi “diamonds” veri setindeki “table” sütununu inceleyelim. NaN değerler düşürüldükten sonraki boxplot ve histogram grafikleri aşağıda sunulmuştur.

![gorsel 2](https://miro.medium.com/max/875/1*ZzC-SUD623uEoX7hTtGDrA.png)

Yukarıdaki grafiklerden boxplotta rahatlıkla görüldüğü üzere, “table” sütununda outlier değerler bulunmaktadır. Ve ayrıca histogram grafiğinden görüldüğü üzere de sütunumuz sağa kuyrukludur (right-skewed).


Outlierların işleme alınmasından önce, nasıl tespit edildiği konusunda fikir sahibi olmakta fayda vardır. Buna IQR kuralı diyebiliriz. Öncelikle verimizdeki çeyrekler arası mesafeyi (Interquartile Range) bulmak gerekmektedir. (IQR = Q3 — Q1). Daha sonra da IQR sınırlarının tespit edilmesi gerekmektedir.


Alt sınırın tespiti için Q1–1.5 * IQR formülünü kullanıyoruz. Aynı şekilde üst sınırın tespiti için Q3 + 1.5 * IQR formülünü kullanıyoruz.


Belirtilen formüle göre, sınırlar arasında kalan data, kabul edilebilir data olarak tanımlanmaktadır. Sınırların dışında kalan data ise outlier olarak değerlendirilmektedir. Tabii ki sınırların tespiti esnasında kullanılan sabit 1.5 değerini 2 veya 2.5 olarak kullanmak da bizim elimizdedir. Buna elimizdeki verinin durumunu değerlendirerek karar vermek uygun bir hal tarzı olacaktır.

![gorsel 3](https://miro.medium.com/max/875/1*jHijFMuswcgEruUZEK4tQg.png)

Pandas’da bulunan describe() metodu ile Q1 (%25) ve Q3(%75) değerlerinin tespiti kolaylıkla yapılabilmektedir. Ve dolayısıyla IQR noktalarının ve sınırların tespiti yapılabilmektedir.

![gorsel 4](https://miro.medium.com/max/875/1*wwdOhCihhK8mXwMuuZb_Rw.png)

Burada bizim üst sınırımız 63.5 ve alt sınırımız 51.5 olarak tespit edilmiştir. Bu da 63.5 ve 51.5 arasındaki verinin kabul edilebilir olduğu anlamına gelmektedir. Yani diğer bir deyişle da 63.5 ve 51.5 arasında kalmayan örnekler outlier olarak tanımlanacaktır. Burada belirtilen outlier değerler ile bir dizi işlemler yapmamız gerekecektir çünkü bahse konu outlier değerler analizde karşımıza beklenmedik sonuçlar ortaya çıkarabilir.


Hadi şimdi bu outlierlar ile başa çıkmaya çalışalım.

#### 1. Outlier değerlerin silinmesi(düşürülmesi)
Kolaylıkla outlier değerleri veri setinden düşürebiliriz. Ancak şunun farkında olmakta fayda var ki bu durum datamızın örneklem sayısını azaltacaktır. Yani outlier değeri düşürmek (silmek) demek, outlier olan bütün satırı düşürmek anlamına gelir. Eğer ki o satırda değerli veriler var ise, o satırda bulunan değerli veriler de outlier ile beraber silinecektir.

![gorsel 5](https://miro.medium.com/max/875/1*j_3a_P89hbIysZ5oguQXcQ.png)

Outlier değerlere ve indekslere ulaşmak için bir dizi işlemlerde bulunduk. Outlierların düşürülmesi, veri setinden 605 kaydın silinmesi anlamına geldi. Hücre 27’de outlierlar görülmektedir. Hücre 28’de ise outlier olmayan data görülmektedir. Toplam örneklem sayımız 53940dan 53335’e düşmüştür.

![gorsel 6](https://miro.medium.com/max/875/1*z3hOsJzLxUmgg_oGLvNgSg.png)

Outlierlar düşürüldükten sonra, yukarıda baktığımız boxplot ve histogram grafiklerini tekrar incelemekte fayda vardır.

![gorsel 7](https://miro.medium.com/max/875/1*USZg5fcbSRW0_MllN5Vx3g.png)

Grafiklerden görüldüğü üzere herhangi bir outlier kalmadı.

#### 2. Winsorize Yöntemi
İkinci metot olarak Winsorize yöntemini kullanabiliriz. Bu yöntemde, outlier değerlerimizi; üst limit ve alt limit ile sınırlandırma yoluna gidiyoruz. Yani sütunda bulunan maksimum değer, üst limitten fazla olamaz, minimum değerde alt limitten az olamaz. Dolayısıyla bütün datayı üst limit, alt limit arasında sıkıştırmış oluyoruz.


Burada yapacağımız işlemlere yine aynı veri seti ile devam edelim. Veri setimiz Seaborn kütüphanesinde bulunan “diamonds” veri seti ve sütunumuzda “table” sütunudur.

![gorsel 8](https://miro.medium.com/max/875/1*hgjJSK0bSvjJusOG9s-VGw.png)

Yukarıda outlier tespiti yapmıştık ve üst limit olarak 63.5, alt limit olarak 51.5 değerlerine ulaşmıştık.


Winsorize yöntemi için Scipy kütüphanesinden winsorize’yi import edelim. Winsorize metodunda kullanmak üzere sınırlara ihtiyacımız olacak. Burada datamızı 63 ve 53 ile limitleyelim. Buradaki değerler outlier limit değerleridir. Yüzdelik olarak bu değerlerin kesin noktalarına ihtiyacımız var ve Pandas’da quantile() metodunu bunun için kullanabiliriz.

![gorsel 9](https://miro.medium.com/max/875/1*DP3VaJU92NY4xI-ko3iX7g.png)

Burada Winsorize yöntemi ile yeni bir değişken tanımlayalım. Yöntemi uygulamak için yüzdelik dilimlerin tam değerlerini yazmam gerekecektir. Yani (0.01, 0.02). Bunun anlamı quantile(0.01) ve quantile(0.98)’i sınır değerler olarak uygulamak istiyorum. İlk değer, sınırın başlangıç noktası, ikinci değer ise sınırın bitiş noktası olarak tanımlanabilir.

![gorsel 10](https://miro.medium.com/max/875/1*kCqMbE0S3PQZcg3dw-YKfQ.png)

Winsorize metodunu uyguladık. Şimdi tekrardan hep beraber boxplot ve histogram grafiklerimizi inceleyelim.

![gorsel 11](https://miro.medium.com/max/875/1*aM5wvx351gq9_R4WN76UBQ.png)

Görüldüğü üzere, outlier bulunmamaktadır. Grafiğe dikkatlice bakıldığında maksimum ve minimum değerlerin 63 ve 53 olduğu görülecektir. Biz bu değerleri sınırlar olarak tanımlamıştık zaten. Şimdi eski ve yeni datanın istatistiksel değerlerine bakalım.

![gorsel 12](https://miro.medium.com/max/875/1*CApjnuKAPnmT620PtlukMA.png)

Görüldüğü üzere ortalama ve standart sapma değerleri değişti. Minimum ve maksimum değerlerde değişti. Ancak bu değişimler medyan tabanlı istatistiksel ifadeleri değiştirmedi. Dolayısıyla bu yöntem uygulanırken de dikkatli olmakta fayda var. Çünkü datanın bozulmasına veya hasar görmesine dolayısıyla analizin ve modelin doğru sonuç vermemesine sebebiyet verilebilir.

#### 3. Log Transformasyonu
Son yöntem olarak Log transformasyonunu kullanalım. Log transformasyonunu kuyruklu (skewed) data üzerinde uyguluyoruz. Log transformasyonu kuyruğu azaltmakta ve normale yakın bir grafik oluşturmaktadır. Bazı durumlarda ise Log transformasyonu normal yapmaktan ziyade daha da kuyruklu bir grafik ortaya çıkarmaktadır. Dolayısıyla Log transformasyonunu uygulamak için en önemli kriter datanın tipidir. Şimdi dataya uygulayalım ve sonucunu gözlemleyelim.
Bu yöntemde, “diamonds” veri setinin “carat” sütununu kullanalım. Grafiklerini inceleyelim.

![gorsel 13](https://miro.medium.com/max/875/1*kacf9m4vDkbw0oipbnWhBw.png)

Veride çok fazla outlier var ve sağa kuyruklu (right-skewed). Log transformasyonu uygulayarak datayı normal veya normale yakın bir grafik haline getirmeyi deneyelim.

![gorsel 14](https://miro.medium.com/max/875/1*RfYTpWp6sSiWUKXkZFE2Cg.png)

NumPy kütüphanesinden np.log() fonksiyonu ile Log transformasyonu uyguladık. Datamız tamamen değişti ve outlierlardan arındı. Bu durumu boxplotta rahatlıkla gözlemlemekteyiz.
Log transformasyonu makine öğrenmesi algoritmalarından sıklıkla kullanılan bir yöntemdir. Log transformasyonu uygulandıktan sonra data değişti ve outlier değerler kayboldu, data normal dağılıma yakın bir hal aldı ve makine öğrenmesi algoritmaları normal dağılımı sever. Makine öğrenmesi algoritmalarında ölçekleme ve normalizasyon gibi farklı algoritmalarda kullanılmaktadır. Diğer yazılarımızda bu konulara da değineceğiz.


#### Sonuç


Outlierlar ile mücadele uzun zaman alsa da buna değdi. Data için en iyi yöntemi seçmek durumundayız. Datayı çok iyi tanımak burada oldukça önemli bir yer tutmaktadır. Burada öğrendiğimiz yöntemler ışığında, artık datamıza hangi yöntemi uygulayarak outlierdan kurtulacağımızı biliyoruz. Bu yöntemlerin dışında, outlier değerleri Nan olarak değerlendirip, Nan değerlere yapılan uygulamaları da burada uygulayabiliriz.
Not: Bu yazı [buradan](https://hersanyagci.medium.com/detecting-and-handling-outliers-with-pandas-7adbfcd5cad8) alınarak Türkçe'ye çevrilmiştir. Bu yazının hazırlanmasında emeği geçen
[Hasan Ersan YAĞCI](https://medium.com/u/7c248ad78b88?source=post_page-----b50e9d3f5178--------------------------------)
’ya sonsuz teşekkürü bir borç bilirim.
