---
layout: post
title:  "Temel Bileşen Analizi (PCA)"
date:   2021-06-20 12:33:07 -0800
categories: yazi
---

![Yosemite - El Capitan](https://miro.medium.com/max/1400/1*HM39uRqzrcGP_6iYd_m73Q.jpeg)

Öz Nitelikleri daha küçük bir alt kümeye ayrıştırmak için birçok teknik vardır. Bu yöntemler, keşif amaçlı veri analizi, görselleştirme, tahmine dayalı modeller oluşturma veya kümeleme problemleri için oldukça faydalı olabilmektedir. Bu yazıda, boyut azaltma için Temel Bileşen Analizini (PCA) ele alacağız. Yazıda, hemen her yerde bulabileceğiniz kod örneklerinden ziyade konunun matematiksel temellerine değinilmiştir.
Temel Bileşen Analizi altında yatan ana fikir, birbiriyle ilişkili birçok değişkenden oluşan bir veri setinin boyutsallığını, veri setinde mevcut varyasyonu maksimum ölçüde korurken azaltmaktır. Orijinal verilerin aynı sayıda veya daha az boyutta bir projeksiyonunu hesaplamak için lineer cebir ve istatistiksel bazı basit matris işlemlerini kullanan bir yöntemdir.

Uygulamada, ister sınıflandırma ister regresyon problemi olsun, bilgi içerdiğini düşündüğümüz gözlem verileri girdi olarak alınır. İdealde, sınıflandırıcı veya regresör, bilgi içermeyen öz nitelikleri modelden atarak gerekli olan özellikleri kullanabilmelidir dolayısıyla ayrı bir süreç olarak öz nitelik seçimine veya çıkarımına ihtiyaç duymamamız gerekir. Ancak, ayrı bir ön işleme adımı olarak boyutluluğu azaltmakla ilgilenmemizin birkaç nedeni bulunmaktadır. Bunlar:

- *Çoğu öğrenme algoritmasında, karmaşıklık, girdi boyutlarının (d) sayısına ve ayrıca veri örneğinin (N) boyutuna bağlıdır, az maliyetli bellek ve hesaplama için, problemin boyutluluğunu azaltmamız gerekmektedir. Girdi boyutunun (d) azaltılması, test sırasında çıkarım algoritmasının karmaşıklığının da azalmasına,*
- *Bir girdinin gereksiz olduğuna karar verildiğinde, onu çıkarma maliyetinden tasarruf edilmesine,*
- *Verilerin daha az öznitelikle açıklanabildiğinde, verinin altında yatan süreç hakkında daha iyi bir fikir edinerek, bilgi çıkarımı yapabilmemize,*
- *Verilerin, bilgi kaybı olmadan birkaç boyutta temsil edilebilmesi, yapı ve aykırı değerler için görsel olarak çizilebilir ve analiz edilebilir hale gelmesine olanak tanımaktadır.*

![Soldaki grafik orijinal verimiz X’i; Sağdaki grafik ise dönüştürülmüş verimizi temsil eder](https://miro.medium.com/max/756/0*5ITfn8msilGwQ-lN.png)

Boyutluluğu azaltmak için iki ana yöntem vardır: öznitelik seçimi ve öznitelik çıkarma.

- *Öznitelik seçiminde, bize en fazla bilgiyi veren d boyuttan k tanesini bulmakla ilgileniyor ve diğer (d-k) boyutları atıyoruz.*
- *Öznitelik çıkarmada ise, orijinal d boyutlarının kombinasyonları olan yeni bir k boyutu kümesi bulmakla ilgileniyoruz. Bu yöntemler, çıktı bilgilerini kullanıp kullanmamalarına bağlı olarak denetimli veya denetimsiz olabilir. En iyi bilinen ve en yaygın olarak kullanılan öznitelik çıkarma yöntemleri, sırasıyla denetimsiz ve denetimli doğrusal iz düşüm yöntemleri olan Temel Bileşenler Analizi (PCA) ve Doğrusal Ayırıcı Analizdir (LDA).*

## Matematiksel Arka Plan

Dağılım ölçümlerine veya verilerin nasıl dağıldığına ilişkin istatistikler ile matrislerin matematiksel arka planları, öz vektör, öz değerler ve matrislerin PCA için temel olan önemli özellikleri üzerine GitHub repoma bir bölüm ekledim. Konu ile ilgili detaylı bilgiyi bu bağlantıyı kullanarak ulaşabilirsiniz.

PCA; verilerdeki paterni tanımanın, verileri benzerliklerini ve farklılıklarını vurgulayacak şekilde ifade etmenin bir yoludur. Verilerin paternini yüksek boyutlu veri setlerinde bulmak zor olabileceğinden, PCA, bizlere verileri analiz etmek için güçlü bir araç sunmaktadır. PCA’nın diğer bir ana avantajı, verilerde bu paterni bulduktan sonra, çok fazla bilgi kaybı olmadan boyut sayısını azaltarak verileri sıkıştırmanızdır.

Hadi bunu bir örnek ile açıklayalım: Bize altı farklı öznitelikte derecelendirilmiş bir müşteri sınıfı verildiğini ve bu müşterileri derecelerine göre sıralamak istediğimizi varsayalım. Kısacası, veri noktaları arasındaki fark en belirgin hale gelecek şekilde verileri tek bir boyuta yansıtmak istiyoruz. Bu noktada PCA’yı kullanabiliriz. Çünkü, en yüksek öz değere sahip öz vektör (en yüksek varyansa sahip yöndür), müşterilerin en çok yayıldığı ekseni temsil eder. Bu durum, temel bileşenler ile temsil edilen varyansı en yüksek seviyede tutacak şekilde veri boyutunu indirgeyebilmemize olanak sağlayacaktır.

Uygulamada, verilerin sahip olduğu özniteliklere ait bazı öz değerlerin açıklanan varyansa çok az katkısı olduğunu ve bunların atılabileceğini anlıyoruz. Örneğin açıklanan varyansın yüzde 95'inden fazlasını açıklayan k tane bileşeni dikkate aldığımızda, λi’ler büyükten küçüğe sıralandığında, k temel bileşenler tarafından açıklanan varyans oranı:

>(λ 1 + λ 2 + … + λ k ) / (λ 1 + λ 2 + … + λ k+ .. + λ d)

Eğer verimize ait öznitelikler aralarında yüksek oranda ilişkiliyse, PCA işlemi sonrasında büyük öz değerleri olan az sayıda öz vektör elde etmiş olacağız bu durumda k sayısı, d’den çok daha küçük olacaktır ve bu sayede boyutlulukta büyük bir azalma elde edilebilecektir. Bu, tipik olarak, yakındaki girdilerin (uzay veya zaman olarak) yüksek oranda ilişkili olduğu birçok görüntü ve ses işleme problemi için geçerlidir. Boyutlar ilişkili değilse, k, d kadar büyük olacaktır ve PCA yoluyla boyut indirgemede kazanç olmayacaktır.

### Adımlar
Şimdi PCA uygulamamız için izlenmesi gereken adımları teker teker inceleyelim.

#### Adım 1: Veriyi Normalize Etmek
İlk adım, PCA’nın düzgün çalışması için elimizdeki verileri normalleştirmektir, burada her bir veri boyutundan ortalamayı çıkarmanız gerekir. Çıkarılan ortalama, her boyuttaki ortalamadır. Böylece, tüm x değerlerinin xb’si (tüm veri noktalarının x değerlerinin ortalaması) çıkarılmış ve tüm y değerlerinin yb’si onlardan çıkarılmıştır. Bu, ortalaması sıfır olan bir veri seti üretir.

#### Adım 2: Kovaryans Matrisini Hesaplayın
Bu adımda yapılacak işlemler yukarıda belirttiğim GitHub repomda detaylı şekilde anlatılmıştır. Buradaki yöntemler incelendiğinde kovaryans matrisinin nasıl hesaplanabileceği görülmüş olacaktır.

#### Adım 3: Öz değer ve Öz vektör Hesaplama
Bu adımda yapmamız gereken kovaryans matrisine ait öz değer ve öz vektörleri hesaplamaktır. ƛ, karakteristik denklemin bir çözümü ise, bir A matrisi için bir öz değeri temsil eder:

>det(Α-λ I)=0

Burada, I matris çıkarma işlemi için gerekli bir koşul olan A ile aynı boyuttaki birim matristir ve bu durumda “det” matrisin determinantıdır. Her bir
öz değer ƛ için, karşılık gelen bir öz vektör v, aşağıdaki eşitliğin çözülmesi ile bulunabilir:

>Αν=λν

#### Adım 4: Bileşenleri Seçme ve Öznitelik Vektörü Oluşturma
Bu adımda ise bir önceki adımda hesapladığımız Öz değerleri büyükten küçüğe sıralarız, böylece bileşenleri sıralı veya anlamlı şekilde elde etmiş oluruz. İşte boyut indirgeme kısmı burada devreye giriyor. d değişkenli bir veri kümemiz varsa, buna karşılık gelen d öz değer ve öz vektöre sahibiz demektir. Boyutları küçültmek için ilk k öz değeri seçip gerisini yok sayarız. Süreçte bazı bilgileri kaybedebiliriz, ancak yok saydığımız bu öz değerler küçükse bu kayıp ihmal edilebilir bir kayıp olacaktır.
Daha sonra, bu öz vektörlerden oluşan bir vektör matrisi oluştururuz. Aslında, bu sadece devam etmek istediğimiz öz vektörlerdir.

>Feature Vector = (eig1, eig2, …)

#### Adım 5: Yeni Veri Setini Türetme
Buraya kadar yaptığımız tüm matematiği kullanarak temel bileşenleri oluşturduğumuz son adıma geldik. Bu adımda, öznitelik vektörünün transpozu ile orijinal verimizin ölçeklendirilmiş halinin transpozunu çarpıyoruz.

>New Data = Feature Vector^T x Scaled Data^T

Bir matrisin tüm öz vektörleri birbirine diktir. Dolayısıyla, PCA’da yaptığımız şey, normal x ve y eksenlerinde temsil etmek yerine bu ortogonal (dikey) öz vektörleri kullanarak orijinal veri setini temsil etmek veya dönüştürmektir.

## Sonuç
Umarım bu yazımı faydalı bulmuşsunuzdur. PCA konusunda daha detaylı bilgi için aşağıdaki referansları takip edebilirsiniz. İyi ve sağlıklı günler dilerim. 😊


$x^2=2$

## Kaynakça
- [1] Alpaydın, E. 2010. “Introduction to Machine Learning.”The MIT Press Cambridge, Massachusetts, London, England
- [2] Smith, L. 2002. “A Tutorial on Principal Components Analysis”
- [3] [https://www.dezyre.com/data-science-in-python-tutorial/principal-component-analysis-tutorial](https://www.dezyre.com/data-science-in-python-tutorial/principal-component-analysis-tutorial)
- [4] [https://www.projectrhea.org/rhea/index.php/PCA_Theory_Examples](https://www.projectrhea.org/rhea/index.php/PCA_Theory_Examples)
- [5] [https://images.app.goo.gl/gYxvW9GAVtCv5uzt8](https://images.app.goo.gl/gYxvW9GAVtCv5uzt8)
