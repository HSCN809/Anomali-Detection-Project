# Anomaly Detection Project

Bu proje synthetic kredi kartı dolandırıcılığı verisi üretir, bu veri üzerinde preprocessing ve EDA çıktıları oluşturur, ardından SVM tabanlı fraud/anomaly modellerini eğitir ve değerlendirir.

Varsayılan çalışma synthetic veri seti içindir. Root klasörde tek komutla tüm pipeline çalıştırılabilir:

```powershell
python main.py
```

## Proje yapısı

```text
DataSet/
  generate_synthetic_fraud_dataset.py
  analysis_script.py

PreProcessing/
  data_cleaning.py
  data_transformation.py
  data_reduction.py
  data_scaling.py
  dataset_statistical_analysis.py

EDA/
  eda_analysis.py

Model Training/
  train_unsupervised_svm.py
  evaluate_unsupervised_svm.py

main.py
requirements.txt
```

## Ortam kurulumu

Projeyi çalıştırmak için Python ve `requirements.txt` içindeki kütüphaneler gerekir. `main.py` bunu otomatik kontrol eder.

Manuel kurulum yapmak istersen:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ana pipeline komutu:

```powershell
python main.py
```

`.venv` zaten aktifse setup kontrolünü atlayabilirsin:

```powershell
.\.venv\Scripts\python.exe main.py --skip-setup
```

## Pipeline özeti

Varsayılan pipeline şu akışı izler:

1. Synthetic fraud veri seti üretir.
2. Eksik değerleri analiz eder ve temizler.
3. Zaman, yaş, mesafe ve kategorik encoding feature'ları üretir.
4. Model için gereksiz veya yüksek cardinality kolonları düşürür.
5. Veriyi scale eder ve train/test olarak ayırır.
6. Statistical analysis tabloları ve PNG çıktıları üretir.
7. EDA grafiklerini üretir.
8. SVM modelini eğitir.
9. Test setinde modeli değerlendirir.

Varsayılan veri boyutu:

```text
Synthetic data: 100,000 rows
Scaling sample: 100,000 rows
Train split: 80,000 rows
Test split: 20,000 rows
One-Class SVM fit: train setindeki non-fraud satırlar, yaklaşık 76,000 rows
```

## 1. Synthetic veri seti üretimi

`DataSet/generate_synthetic_fraud_dataset.py` Faker tabanlı synthetic transaction datası üretir. Veri seti gerçek fraud datasına benzer kolon adları kullanır ve `is_fraud` hedef kolonunu içerir.

Varsayılan çıktı:

```text
DataSet/synthetic_fraud_merged.csv
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe DataSet\generate_synthetic_fraud_dataset.py
```

Argümanlar:

- `--output`: Üretilecek CSV dosyasının yolu.
- `--rows`: Üretilecek satır sayısı. Varsayılan `100000`.
- `--fraud-rate`: Fraud oranı. Varsayılan `0.05`.
- `--random-state`: Tekrarlanabilir random üretim için seed.

Örnek:

```powershell
.\.venv\Scripts\python.exe DataSet\generate_synthetic_fraud_dataset.py --rows 100000 --fraud-rate 0.05
```

Teknik olarak bu adım transaction zamanı, müşteri bilgileri, merchant/category bilgisi, amount, lokasyon, fraud etiketi ve `source_dataset` kolonlarını üretir.

## 2. Data cleaning

`PreProcessing/data_cleaning.py` synthetic merged dosyasındaki eksik değerleri analiz eder, missing value tablosunu PNG olarak export eder ve temizlenmiş dataset üretir.

Varsayılan input:

```text
DataSet/synthetic_fraud_merged.csv
```

Varsayılan output:

```text
DataSet/synthetic_fraud_cleaned.csv
PreProcessing/synthetic_prep_outputs/missing_value_analysis.png
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe PreProcessing\data_cleaning.py
```

Argümanlar:

- `--input`: Temizlenecek input CSV yolu.
- `--output`: Temizlenmiş CSV output yolu.
- `--output-dir`: PNG analiz çıktılarının yazılacağı klasör.
- `--nrows`: Sadece belirli sayıda satır okumak için kullanılır.

Teknik olarak numeric kolonlarda median imputation, categorical kolonlarda mode veya `"Unknown"` imputation uygular.

## 3. Data transformation

`PreProcessing/data_transformation.py` cleaned dataset üzerinde modelleme için yeni feature'lar üretir.

Varsayılan input:

```text
DataSet/synthetic_fraud_cleaned.csv
```

Varsayılan output:

```text
DataSet/synthetic_fraud_transformed.csv
PreProcessing/synthetic_prep_outputs/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe PreProcessing\data_transformation.py
```

Argümanlar:

- `--input`: Cleaned CSV yolu.
- `--output`: Transformed CSV output yolu.
- `--output-dir`: Transformation summary ve grafik çıktılarının klasörü.

Üretilen başlıca feature'lar:

- `transaction_hour`
- `transaction_day_of_week`
- `transaction_day_of_month`
- `transaction_month`
- `is_weekend`
- `is_night`
- `customer_age`
- `customer_merchant_distance_km`
- `category_*` one-hot kolonları
- `gender_*` one-hot kolonları

Teknik olarak timestamp alanından zaman feature'ları, `dob` alanından yaş, müşteri ve merchant koordinatlarından Haversine mesafesi hesaplanır.

## 4. Data reduction

`PreProcessing/data_reduction.py` model eğitiminde doğrudan kullanılmayacak raw veya yüksek cardinality kolonları düşürür.

Varsayılan input:

```text
DataSet/synthetic_fraud_transformed.csv
```

Varsayılan output:

```text
DataSet/synthetic_fraud_transformed_reducted.csv
PreProcessing/synthetic_prep_outputs/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe PreProcessing\data_reduction.py
```

Argümanlar:

- `--input`: Transformed CSV yolu.
- `--output`: Reduced CSV output yolu.
- `--output-dir`: Reduction summary PNG çıktılarının klasörü.

Düşürülen başlıca kolonlar:

- Raw timestamp ve unix time kolonları
- Kart ve kişi kimlik bilgileri
- Raw adres/lokasyon kolonları
- `merchant`, `first`, `last`, `street`, `city`, `state`
- `trans_num`, `source_dataset`

Teknik olarak model için daha kompakt bir tablo bırakır; `job` ve `zip` kolonları frequency encoding için training adımına kadar tutulur.

## 5. Data scaling ve train/test split

`PreProcessing/data_scaling.py` reduced dataset üzerinde sampling, cyclic encoding, train/test split ve scaling uygular.

Varsayılan input:

```text
DataSet/synthetic_fraud_transformed_reducted.csv
```

Varsayılan output:

```text
DataSet/synthetic_fraud_transformed_reducted_sampled.csv
DataSet/synthetic_fraud_transformed_reducted_scaled_train.csv
DataSet/synthetic_fraud_transformed_reducted_scaled_test.csv
PreProcessing/synthetic_prep_outputs/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe PreProcessing\data_scaling.py
```

Argümanlar:

- `--input`: Reduced CSV yolu.
- `--sample-output`: Sampled CSV output yolu.
- `--train-output`: Scaled train CSV output yolu.
- `--test-output`: Scaled test CSV output yolu.
- `--output-dir`: Scaling summary ve grafik çıktılarının klasörü.
- `--sample-size`: Sampling yapılacak satır sayısı.
- `--fraud-sample-rate`: Sample içindeki fraud oranı.
- `--no-sample`: Sampling yapmadan tüm input dataset ile devam eder.

Teknik işlemler:

- `transaction_hour`, `transaction_day_of_week`, `transaction_month` için sine/cosine cyclic encoding üretir.
- `amt`, `city_pop`, `customer_merchant_distance_km` kolonlarına `log1p + RobustScaler` uygular.
- `customer_age` için `StandardScaler` uygular.
- `transaction_day_of_month` için `MinMaxScaler` uygular.
- Stratified train/test split kullanır.

Varsayılan main pipeline’da sample size `100000` olduğu için generated synthetic datasetin tamamı kullanılır.

## 6. Statistical dataset analysis

`PreProcessing/dataset_statistical_analysis.py` synthetic merged dataset için genel istatistik ve tablo çıktıları üretir.

Varsayılan input:

```text
DataSet/synthetic_fraud_merged.csv
```

Varsayılan output:

```text
PreProcessing/synthetic_prep_outputs/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe PreProcessing\dataset_statistical_analysis.py
```

Bu script argüman almaz; default synthetic merged dosyasını okur.

Ürettiği analizler:

- Genel satır/kolon bilgisi
- Kolon listesi
- Data type dağılımı
- `head()`
- `describe()`
- Missing value özeti
- Duplicate row sayısı
- Target distribution
- Numeric column summary
- Categorical cardinality
- Kategorik kolonlar için top values

## 7. EDA analysis

`EDA/eda_analysis.py` transformed dataset üzerinden görsel EDA çıktıları üretir.

Varsayılan input:

```text
DataSet/synthetic_fraud_transformed.csv
```

Varsayılan output:

```text
EDA/synthetic_eda_outputs/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe EDA\eda_analysis.py
```

Argümanlar:

- `--input`: Transformed CSV yolu.
- `--output-dir`: EDA PNG çıktılarının klasörü.

Ürettiği başlıca grafikler:

- Target distribution
- Amount distribution
- Log amount distribution
- Amount by fraud status boxplot
- Category fraud rate
- Category transaction/fraud counts
- Distance distribution
- Distance by fraud status
- Distance bucket fraud rate
- Amount vs distance scatter sample
- Hourly fraud rate
- Night flag fraud rate
- Feature correlation heatmap

## 8. Model training

`Model Training/train_unsupervised_svm.py` scaled train dataset üzerinde SVM tabanlı modelleri eğitir.

Varsayılan input:

```text
DataSet/synthetic_fraud_transformed_reducted_scaled_train.csv
```

Varsayılan output:

```text
Model Training/models/synthetic_oneclass_rbf_svm_model.pkl
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe "Model Training\train_unsupervised_svm.py"
```

Argümanlar:

- `--input`: Scaled train CSV yolu.
- `--model-path`: Linear supervised SVM model artifact yolu.
- `--kernel-model-path`: RBF supervised SVM model artifact yolu.
- `--oneclass-model-path`: One-Class RBF SVM model artifact yolu.
- `--training-sample-size`: Genel training sample satır sayısı.
- `--kernel-training-sample-size`: Kernel SVM için sample satır sayısı.
- `--oneclass-training-sample-size`: One-Class SVM için sample satır sayısı. `0` veya verilmemesi tüm train verisini kullanır.
- `--validation-size`: Threshold tuning validation split oranı.
- `--chunksize`: CSV chunk okuma boyutu.
- `--random-state`: Reproducible split ve sampling seed değeri.
- `--models`: Eğitilecek modeller. Seçenekler: `linear`, `kernel`, `oneclass`, `both`, `all`.
- `--param-preset`: Hyperparameter grid seçimi. Seçenekler: `fast`, `default`.
- `--dry-run`: Modeli fit eder ama artifact kaydetmez.

Varsayılan `main.py` akışında:

```text
--models oneclass
--param-preset fast
--oneclass-training-sample-size 0
```

Bu ayar 100.000 synthetic satırdan gelen 80.000 train satırını kullanır ve One-Class SVM fit işlemini train setindeki non-fraud satırlarda yapar.

## 9. Model evaluation

`Model Training/evaluate_unsupervised_svm.py` trained model artifactlerini scaled test set üzerinde değerlendirir ve metrik/grafik çıktıları üretir.

Varsayılan input:

```text
DataSet/synthetic_fraud_transformed_reducted_scaled_train.csv
DataSet/synthetic_fraud_transformed_reducted_scaled_test.csv
DataSet/synthetic_fraud_transformed_reducted.csv
DataSet/synthetic_fraud_transformed_reducted_sampled.csv
Model Training/models/synthetic_oneclass_rbf_svm_model.pkl
```

Varsayılan output:

```text
Model Training/evaluation_outputs_synthetic/
```

Tek başına çalıştırma:

```powershell
.\.venv\Scripts\python.exe "Model Training\evaluate_unsupervised_svm.py"
```

Argümanlar:

- `--train`: Scaled train CSV yolu. Frequency maps buradan yeniden hesaplanır.
- `--test`: Scaled test CSV yolu.
- `--reduced`: Original amount reconstruction için reduced CSV yolu.
- `--sampled-source`: Scaling öncesi sampled source CSV yolu.
- `--model`: Linear supervised SVM artifact yolu.
- `--kernel-model`: Kernel supervised SVM artifact yolu.
- `--oneclass-model`: One-Class SVM artifact yolu.
- `--models`: Değerlendirilecek model veya modeller. Seçenekler: `linear`, `kernel`, `oneclass`, `both`, `all`.
- `--output-dir`: Evaluation CSV ve PNG çıktılarının klasörü.
- `--chunksize`: CSV chunk okuma boyutu.
- `--eval-sample-size`: Evaluation için opsiyonel sample satır sayısı.
- `--random-state`: Evaluation sampling seed değeri.
- `--inference-sample-size`: Inference latency ölçümü için kullanılacak sample satır sayısı.

Ürettiği metrikler:

- Confusion matrix
- Accuracy
- Recall
- False positive rate
- Precision
- F1 score
- AUPRC
- ROC-AUC
- Predicted fraud rate
- Total fraud amount
- Captured fraud amount
- Fraud loss capture rate
- Inference time per transaction

Ürettiği başlıca çıktılar:

```text
evaluation_summary.csv
threshold_report.csv
evaluation_summary_table.png
confusion_matrix.png
precision_recall_curve.png
roc_curve.png
threshold_metrics.png
fraud_loss_capture_curve.png
```

## Hızlı deneme komutları

Küçük veriyle hızlı smoke test çalıştır:

```powershell
python main.py --rows 10000 --sample-size 10000 --oneclass-training-sample-size 1000
```

Modeli kaydetmeden training logic test et:

```powershell
python main.py --dry-run-training
```

Tüm modelleri eğit ve değerlendir:

```powershell
python main.py --model all --training-preset default
```

Bu komut linear, kernel ve one-class modellerini çalıştırır. Kernel RBF SVM büyük veride daha uzun sürebilir.

## Output dosyaları

Pipeline çıktıları tekrar çalıştırıldığında overwrite edilir. Başlıca output klasörleri:

```text
DataSet/*.csv
PreProcessing/synthetic_prep_outputs/
EDA/synthetic_eda_outputs/
Model Training/models/
Model Training/evaluation_outputs_synthetic/
```

Bu dosyalar `.gitignore` içinde ignore edilir.

## main.py

`main.py` proje için ana orkestrasyon dosyasıdır. Tek komutla environment setup, veri üretimi, preprocessing, EDA, model training ve evaluation adımlarını sırayla çalıştırır.

Çalıştırma:

```powershell
python main.py
```

`main.py` önce environment kontrolü yapar:

- Rootta `.venv` var mı kontrol eder.
- `.venv` yoksa `python -m venv .venv` ile oluşturur.
- `requirements.txt` içindeki paketleri `.venv` içinde `pip show` ile kontrol eder.
- Eksik paket varsa `.venv\Scripts\python.exe -m pip install -r requirements.txt` çalıştırır.
- Sistem Python ile başlatıldıysa pipeline’ı `.venv` Python’u ile yeniden çalıştırır.

`main.py` sonra pipeline adımlarını Rich ile terminalde izlenebilir şekilde çalıştırır:

- Her adım için step başlığı basar.
- Çalıştırılan komutu gösterir.
- Adımın süresini gösterir.
- Beklenen output pathlerini listeler.
- Bir adım hata verirse pipeline’ı durdurur ve failed command bilgisini gösterir.

`main.py` argümanları:

- `--skip-setup`: `.venv` ve dependency kontrolünü atlar.
- `--rows`: Üretilecek synthetic satır sayısı. Varsayılan `100000`.
- `--fraud-rate`: Synthetic fraud oranı. Varsayılan `0.05`.
- `--random-state`: Pipeline random seed değeri. Varsayılan `42`.
- `--sample-size`: Scaling adımında kullanılacak satır sayısı. Varsayılan `100000`, yani tüm generated dataset kullanılır.
- `--model`: Eğitilecek/değerlendirilecek model grubu. Seçenekler: `oneclass`, `linear`, `kernel`, `both`, `all`. Varsayılan `oneclass`.
- `--training-preset`: Training grid preset seçimi. Seçenekler: `fast`, `default`. Varsayılan `fast`.
- `--oneclass-training-sample-size`: One-Class SVM için training sample satır sayısı. Varsayılan `0`, yani tüm train satırları kullanılır.
- `--eval-sample-size`: Evaluation için opsiyonel sample satır sayısı.
- `--dry-run-training`: Training adımını çalıştırır ama model dosyası kaydetmez.

Varsayılan `python main.py` davranışı:

```text
100,000 synthetic row üretir.
100,000 row ile preprocessing/scaling yapar.
80,000 row train ve 20,000 row test split oluşturur.
One-Class RBF SVM modelini train setindeki non-fraud satırlar ile eğitir.
Test set üzerinde modeli değerlendirir.
CSV, PNG ve PKL çıktılarını var olan dosyaların üzerine yazar.
```
