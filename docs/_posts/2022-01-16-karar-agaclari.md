---
layout: post
title:  "Karar Ağaçları (Decision Trees)"
date:   2022-01-16 21:14:37 -0800
categories: yazi
---

***Editörün notu:*** *Sayı sıkıştırmaca oynarken bir sonraki tahmininizi nasıl seçerdiniz? Büyükten küçüğe tüm sayıları söyleyerek mi giderdiniz, yoksa aranacak alanı ortadan bölen tahminlerde mi bulunurdunuz? Seçtiğiniz stratejinin veri bilimindeki karşılığı nedir? Bilgi kazancı ve safsızlık nasıl ölçülür?, [Ahmet Ümit](https://medium.com/@aumitcaliskan)'in tüm bu konulara ve tüm detaylarıyla karar ağaçlarına değindiği yazısı sizlerle...*

*[Ahmet Ümit Çalışkan](https://medium.com/@aumitcaliskan)'ın [Machine Learning Turkiye](https://medium.com/machine-learning-t%C3%BCrkiye)'de 16 Ocak 2022 tarihinde yayınlanmış olan yazısıdır.*

Karar ağaçları (Decision Trees) algoritması, makine öğrenmesinin denetimli öğrenme (Supervised Learning) dalında kullanılan, parametrik olmayan (non-parametric), sınıflandırma (classification) ve regresyon (regression) için kullanılan ağaç bazlı (tree based) bir algoritmadır.

Karar ağaçları algoritmasını incelemeden önce parametrik olmayan ne demektir kısaca ondan bahsedelim. Makine öğrenmesi algoritmalarını temel olarak iki gruba ayırabiliriz: Parametrik olanlar ve olmayanlar. Parametrik olmayan yöntem, veri hakkında herhangi bir varsayımda bulunmayan bir istatistik yöntemidir. Bu parametresi olmayan demek değildir. Parametrelerin daha esnek ve başlangıçta sabit parametrelerin olmadığı anlamına gelmektedir. Varsayım yapılmadığı için parametrik olmayan algoritmalar, eğitim verisinde öğrenirken oldukça rahat ve esnek davranabilirler.

>*“Elinizde büyük bir veri varsa, öncesine ait bir bilgi yoksa ve doğru özellikleri (feature) seçme konusunda çok fazla endişelenmek istemiyorsanız; parametrik olmayan yöntemler iyi bir tercih olacaktır.”*
>
>*— Artificial Intelligence: A Modern Approach, sayfa 757*

En çok bilinen parametrik olmayan makine öğrenmesi algoritmaları şu şekildedir:
- K-En Yakın Komşular (K-Nearest Neighbors)
- Karar Ağaçları (Decision Trees)
- Destek Vektör Makineleri (Support Vector Machines)

Karar ağaçları, datayı özelliklerden aldığı bilgiye göre ayıran bir algoritmadır. Peki karar ağacı nasıl oluşur? Hepimizin bildiği sayı bulma oyununu oynayarak basit bir karar ağacı oluşturalım. 0 ile 20 arasında rasgele bir tam sayı seçelim. Ve bunu tahminlerle bulmaya çalışalım.

```python
def sayi_bul():
    x = np.random.randint(0,21)
    y = int(input("Tahmin edilen sayı: "))
    while y != x:
        if y > x:
            print("Daha küçük bir sayı deneyiniz")
            y = int(input("Tekrardan: "))
        elif y < x:
            print("Daha yüksek bir sayı deneyiniz")
            y = int(input("Tekrardan: "))
        else:
            break
    print("Tebrikler. Sayıyı buldunuz!")
```
```
Tahmin edilen sayı: 10
Daha yüksek bir sayı deneyiniz
Tekrardan: 15
Daha yüksek bir sayı deneyiniz
Tekrardan: 17
Daha yüksek bir sayı deneyiniz
Tekrardan: 19
Tebrikler. Sayıyı buldunuz!
```

![](https://miro.medium.com/max/875/1*bAA4EMWlvnQK4nyUIFVAmA.png)

Şekilden de anlaşılacağı üzere karar ağacı algoritmasının if else gibi çalıştığını söyleyebiliriz. 10 sayısını girdikten sonra daha yüksek bir tahminde bulunmamız istendi. Bu durumda rasgele seçilen sayı 10 dan büyük. Daha sonra 15 sayısını denedik ve yine daha yüksek bir tahminde bulunmamız istendi. Demek ki 15"ten de büyük. Bu işlem sayıyı bulana kadar devam etti. Eğer 10 dan büyük olmasaydı bu sefer 5 girip 5"ten büyük mü diye sorgulayacaktık ve aynı mantıkla sayıyı tahmin etmeye çalışacaktık.

Karar ağacının terimlerine değinecek olursak; ağacın başladığı turuncu ile gösterilen ilk kutuyu *“kök düğümü (root node)”*, dallanıp budaklandığı kutuları *“düğüm (node)”* ve ağacın en uç noktasında kalan artık dallanamayan yeşil kutucukları ise *“yaprak düğümü (leaf(terminal) node)”* olarak adlandırıyoruz. Ayrıca dallara ayrılan düğümlere “ebeveyn (parent node)”, dallara ayrılmayan ancak bir dala bağlı olan düğümlere ise “çocuklar (children nodes)”, ağacın içinde gördüğümüz kırmızı çember içerisine alınan bir ebeveyn ve çocuklardan oluşan kısma da *“alt ağaç (sub-tree)”* diyoruz.

---

Peki karar ağacı algoritması nasıl çalışıyor? Hangi özelliği neye göre seçip kök olarak alıyor ve dallanmaya başlıyor? Hangi parametreleri kullanıyor? Şimdi de bu soruların cevaplarını arayalım.

Algoritmanın amacı; özellikleri mümkün olduğunca hızlı ve maliyeti düşük bir şekilde bölmektir. Sayı bulma oyununu hatırlayalım. 0 ile 20 arasında rasgele seçilen bir sayıyı tahmin etmeye çalışırken ilk olarak 10 ile başladık. 1 ile de başlayabilirdik. Peki neden 10 ile başladık? Mümkün olduğunca hızlı ve en kısa şekilde sayıyı bulmaya çalışıyoruz. Hatırlarsanız sayımız 19 çıkmıştı. Eğer biz 1 den başlayıp, birer birer artırma yöntemini seçseydik 19. adımda sayıyı bulacaktık. Ama yukarıda 4 adımda sayıyı bulduk. Sayı kümesinin ortasından başladık ki veriyi önce iki sınıfa ayıralım ve bir sınıfı seçerek devam edelim. Karar ağacı algoritması da aynı şekilde ağacı oluşturmaya başlarken özellikleri inceler ve en saf özelliği seçerek bu özelliği kök yapar. Buradaki saflıktan kastımız verinin dağılımıdır. Saflığı hesaplamak için de iki farklı yöntem kullanılır:

#### 1. Entropi:

Entropi, sınıflandırılan verinin safsızlık ölçüsüdür. 0 ve 1 arası bir değer alır. Amacımız sınıflandırma olduğu için çıktı olarak 0 ya da 1 istiyoruz. Bunun için teorik olarak entropinin 0"a yaklaşmasını istiyoruz ki saflık derecemiz yüksek olsun. Algoritmamız da özellikler arasından en düşük entropiye sahip olanı kök olarak alıyor ve dallandırmaya başlıyor.

![](https://miro.medium.com/max/605/1*en6d0tETjj8Sq8uiL8TAAg.png)
![](https://miro.medium.com/max/555/1*5dejHc3kX5clkMuXvHv5ew.png)

#### 2. Gini Impurity:
Gini impuriti de aynı şekilde verinin safsızlığının ölçüsüdür. Entropinin aksine 0 ve 0.5 arası değer alır.Yapraklarda mümkün olduğunca gini impuriti sayısını minimize etmeye çalışıyoruz.

![](https://miro.medium.com/max/499/1*fVbDXmT677jsau6NeaxUJA.png)
![](https://miro.medium.com/max/418/1*lpnXHlwu7crEQ9eU4F4Tlw.png)

Gini impuriti ile entropinin temel farkı hesaplama kolaylığıdır. Entropi hesaplamasında logaritma kullanıldığı için hesaplaması gini impuritiye göre daha uzun sürebilir. Yani entropi hesaplama maliyeti olarak pahalıdır. (computationally expensive)

Özelliklerden hangisinin kök olacağına karar veren algoritmamız aynı yöntemle düğümler oluşturarak ağacı dallandırır. Bu işlem yaprakları elde edene kadar devam eder.

---

Karar ağaçları algoritması sınıflandırıcı ve regresyon olmak üzere iki class halinde scikit-learn kütüphanesinde bulunmaktadır. Öyleyse hiperparametrelerine göz atalım.

#### Hiperparametreler
- **criterion** : *{“gini”, “entropy”}, default=”gini”*. Yukarıda bahsetmiş olduğumuz saflığı hesaplarken kullanacağımız kriterler.
- **splitter** : *{“best”, “random”}, default=”best”*. Her bir düğümdeki bölünmeyi seçtiğimiz strateji.
- **max_depth** : *int, default=None*. Karar ağacının maksimum derinliği. Derinlik; kök düğümünden yaprak düğümüne kadar olan bölünme sayısını ifade eder. Eğer “None” seçilirse yaprak düğümleri tamamen saf olana kadar düğümler bölünür.
- **min_samples_split** : *int or float, default=2*. Düğümlerin bölünmesi için gereken minimum örnek sayısı. Örneğin; 10 seçilirse düğümde 10"dan az örnek olduğunda düğüm bölünmez.
- **min_samples_leaf** : *int or float, default=1*. Yaprak düğümünde olması gereken minimum örnek sayısı. Herhangi bir değer girilmezse yaprak düğümünde 1 örnek kalana kadar düğümleri bölmeye devam eder.
- **min_weight_fraction_leaf** : *float, default=0.0.* Ağırlıklı örneklerin toplam örnekler içerisindeki oranı.
- **max_features** : int, float or {“auto”, “sqrt”, “log2”}, default=None. En iyi bölünme için düşünülen özellik sayısı. “auto”
  - :max_features=sqrt(n_features), “sqrt”
  - :max_features=sqrt(n_features), “log2”
  - :max_features=log2(n_features), “None”
  - :max_features=n_features
- **max_leaf_nodes** : *int, default=None.* Maksimum yaprak düğümü sayısı. Diyelim ki değer girmedik ve modelimizde 10 tane yaprak oluştu ama biz yaprak sayısını 5 istiyoruz. max_leaf_nodes = 5 değerini girdiğimizde algoritma en yüksek saflık derecesine sahip yani gini ya da entropi değeri en düşük 5 düğümü yaprak olarak bırakır.
- **min_impurity_decrease** : *float, default=0.0*. Bir düğüm bölünürken safsızlık, girilen değere eşit ya da bu değerden büyük azalacaksa yani istediğimiz derecede saflığı elde edecekse bölünür, yoksa bölünmez.

---

Parametreleri de açıkladıktan sonra karar ağaçlarının avantajlarını ve dezavantajlarını sıralayalım.

#### Avantajları
- Anlaması, görselleştirilerek anlaşılması ve yorumlanması kolaydır.
- Verilerin ölçeklendirilmesine gerek yoktur.
- Verilerin normalleştirilmesine gerek yoktur.
- Sayısal verilerin yanında kategorik veriler ile de çalışabilir.
- Diğer algoritmalara göre veri hazırlığı için daha az zaman gerekir.

#### Dezavantajları
- Verideki küçük bir değişiklik ağacın yapısında büyük bir değişikliğe sebep olabilir.
- Hesaplaması diğer algoritmalara göre daha karmaşık olabilir.
- Eğitmek için daha fazla süre gerekir.
- Regresyon uygulaması diğer algoritmalara göre yeterli değildir.
- Son olarak ve en önemlisi ezberleme (over-fitting) olasılığı yüksektir.

Karar ağaçları algoritmasını kullanırken en çok dikkat etmemiz gereken konu ***“ezberleme (over-fitting)”*** olmalıdır. Algoritma yapısı gereği ezberlemeye müsaittir. Düğümler bölüne bölüne yapraklara ulaşır. Eğer hiperparametrelere müdahale etmezsek bir örnek kalana kadar düğümler bölünür. Büyük bir veride sadece eğitim setini gören algoritmamız bu şekilde veriyi ezberleyebilir. Bunun önüne geçmek için ***“budama (pruning)”*** dediğimiz teknik uygulanır. Yukarıda açıklamalarını yaptığımız max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes ve min_impurity_decrease budama için en sık kullanılan hiperparametrelerdir.

---

Buraya kadar konuştuklarımızı örnek bir veri seti üzerinden pekiştirelim. Örnek verimizde penguenlerin türleri, yaşadıkları adalar ve fiziksel özellikleri verilmiştir. Amacımız penguenleri; yaşadıkları adaya ve fiziksel özelliklerine bakarak türlerine göre ayırmak. Algoritmamız sınıflandırma yapacağından, ***DecisionTreeClassifier*** classını kullanacağız. Veri setine ve python dosyasına [buradan](https://github.com/aumitcaliskan/Machine-Learning/tree/main/Decision%20Trees) ulaşabilirsiniz.

![](https://miro.medium.com/max/875/1*zh_aHWNHjkae6qQ9agkuhQ.png)

Verimizi düzenledikten sonra "species" sütununu y, kalan sütunları da X olarak tanımlayarak train ve test olarak ayırıyoruz.

```python
X = pd.get_dummies(df.drop("species",axis=1),drop_first=True)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

Şimdi modelimizi tanımlayalım ve ızgara araması (grid search) uygulayarak en iyi hiperparametreleri bulalım.

```python
model = DecisionTreeClassifier()
parameters = {“max_depth”: range(1,21),
              “min_samples_split”: range(2,21,2),
              “min_samples_leaf”:range(1,21),
              “max_leaf_nodes”: range(2,21),
              “min_impurity_decrease”: np.arange(0,1,0.1)}
grid_model = GridSearchCV(estimator=model,
                          param_grid= parameters,
                          cv=3,return_train_score=True, n_jobs= -1)
grid_model.fit(X_train, y_train)
```
```python
grid_model.best_params_
```
```
{"max_depth": 4,"max_leaf_nodes": 9,"min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2}
```
```python
grid_model.best_score_
```
```
0.9743034743034743
```

Izgara aramasından elde ettiğimiz değerlerle kurulu ağacımız ve skorlarımız aşağıda verilmiştir.

![](https://miro.medium.com/max/875/1*gmYLbTs80s4Psr7be9iZGQ.png)

![](https://miro.medium.com/max/699/1*drBcn1AVB6EVun80X9iL-g.png)

Hiperparametrelerin değişiminin eğitim ve test puanları üzerindeki etkilerini görselleştirelim:

![](https://miro.medium.com/max/875/1*rQSGXGPImzsIspk_bwV3AA.png)

Hiperparametrelerin grafiklerini incelediğimizde hangi değerler sonrasında ezberlemeye doğru gidildiği açıkta görülmektedir. max_depth için 5’ten sonrası, max_leaf_nodes için 10’dan sonrası overfite meyilli iken, min_samples_leaf için 10’dan sonrası, min_impurity_decrease için 0.4’ten sonrası ve min_samples_split için de 5’ten sonrası underfite meyillidir.

***Sonuç olarak;*** karar ağaçları algoritması, veri hazırlığının kısa sürmesi, kolay bir şekilde anlaşılması, yorumlanması ve görselleştirilmesi düşünüldüğünde avantajlı, hesaplama açısından pahalı ve overfite meyilli olması nedeniyle kullanılırken dikkat edilmesi gereken bir algoritmadır.
