import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
#Görülen tablo ayarları için
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
"""

# Veri Dosyalarının Yüklenmesi
berlin_file = '../datasets/berlin_era5_wind_20241231_20241231.csv'
munich_file = '../datasets/munich_era5_wind_20241231_20241231.csv'


def load_and_explore_datasets(berlin_file, munich_file):
    try:
        df_b = pd.read_csv(berlin_file)
        df_m = pd.read_csv(munich_file)
    except FileNotFoundError:
        print("Error: File not found.")
        return None, None

    print("=" * 50)
    print("### Data Importing and Basic Informations ###")

    # "timestamp": veriler ne zaman kaydedilmiş onu verir.
    # datetime nesnesi, saat ve tarihin sayısal bir temsilini içerir.
    # df_b = Berlin'in rüzgar verilerini tutan tablo
    df_b['timestamp'] = pd.to_datetime(df_b['timestamp'])
    df_m['timestamp'] = pd.to_datetime(df_m['timestamp'])

    # Temel Bilgilerin Görüntülenmesi (Shape, Columns, Data Types)
    # Name değişkeni, Berlin ve Munich olur sırayla. df ise kendi veri tabloları olur.
    for (name, df) in ("BERLIN", df_b), ("MUNICH", df_m):
        print(f"\n--- {name} Basic Informations ---")
        print(f"Shape: {df.shape}")

        #Sütunların başlıkları alınıp liste oluşturulur.
        #df.columns, kitaplığın üstündeki etiketleri verir.
        #.tolist() bu etiketleri listeler.
        column_list = df.columns.tolist()
        print(f"Column Count: {len(column_list)}")
        print(f"Column Names: {column_list}")
        #Her sütunun data türünü verir.
        print("Data Types:")
        print(df.dtypes)

        # len() fonksiyonu, data framelerde rowları gösteriyor.
        initial_rows_b = len(df_b)
        initial_rows_m = len(df_m)

        #.dropna komutunda, tablo satır satır incelenir ve eksik hücre varsa satırın tamamı silinir.
        #inplace= True, tablonun içeriğini kalıcı olarak siler ve değiştirir.
        df_b.dropna(inplace=True)
        df_m.dropna(inplace=True)

        #inplace = true dediği için orijinal dosyayı değiştiriyor.

        dropped_rows_b = initial_rows_b - len(df_b)
        dropped_rows_m = initial_rows_m - len(df_m)

        print("\n--- Missing Value Handling ---")
        print(f"BERLIN: Dropped {dropped_rows_b} rows (Remaining: {len(df_b)} rows)")
        print(f"MUNICH: Dropped {dropped_rows_m} rows (Remaining: {len(df_m)} rows)")

        print("\n--- BERLIN Summary Statistics ---")
        #.describe() komutu, özet istatistik çıkarır.
        # %25 = first quartile
        # %50 = median
        # %75 = third quartile

        print(df_b[['u10m', 'v10m', 'lat', 'lon']].describe())

        print("\n--- MUNICH Summary Statistics ---")
        print(df_m[['u10m', 'v10m', 'lat', 'lon']].describe())

        print("=" * 50)
        return df_b, df_m

def calculate_wind_speed(df):
    # Hız vektörü bulmak için: sqrt(u^2 + v^2)
    # df'ye büyüklüğün olduğu bir sütun eklenir.
    df['wind_speed'] = np.sqrt(df['u10m']**2 + df['v10m']**2)
    return df

def compute_monthly_averages(df_berlin_raw, df_munich_raw):
    print("\n" + "=" * 50)
    print("### MONTHLY AVERAGES ###")

    #ilk indexleyip sonra kopyasını oluşturuyoruz. deep=true, yeni sütun eklediğimizde orijinali değişmesin diye.
    df_b = df_berlin_raw.set_index('timestamp').copy(deep=True)
    df_m = df_munich_raw.set_index('timestamp').copy(deep=True)

    # rüzgar hızını hesaplama ve yeni sütun ekleme)
    df_b = calculate_wind_speed(df_b)
    df_m = calculate_wind_speed(df_m)

    # 2. Aylık Ortalamaları Hesaplama (Monthly Averages)

    # .resample('M'): Veriyi aylık gruplara ayırır ('M' = Month End).
    # .mean(): Her bir aylık gruptaki tüm saatlik değerlerin ortalamasını alır.
    monthly_avg_b = df_b['wind_speed'].resample('M').mean()
    monthly_avg_m = df_m['wind_speed'].resample('M').mean()

    # Sonuçları Görüntüleme
    print("\n--- Monthly Average Wind Speeds (m/s) ---")
    print("BERLIN:\n", monthly_avg_b.to_string())  # to_string() konsolda temiz gösterir
    print("\nMUNICH:\n", monthly_avg_m.to_string())

    print("=" * 50)

    # Mevsimleri belirleyen yardımcı fonksiyon (Aylar: 12,1,2=Kış; 3,4,5=İlkbahar; 6,7,8=Yaz; 9,10,11=Sonbahar)

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Autumn'

    #map(get_season) ay numaraları listesini alıp ay karşılıklarını veriyor.
    #yeni bir sütunda, verilerin ait olduğu aylar gösterilir.
    df_b['season'] = df_b.index.month.map(get_season)
    df_m['season'] = df_m.index.month.map(get_season)

    #mevsimlerin ortalaması alınır.
    #reindex istediğin sırayla sıralamak için
    SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_avg_b = df_b.groupby('season')['wind_speed'].mean().reindex(SEASON_ORDER)
    seasonal_avg_m = df_m.groupby('season')['wind_speed'].mean().reindex(SEASON_ORDER)

    #Mevsimsel ortalamayı print eder.
    print("\n--- Seasonal Average Wind Speeds (m/s) ---")
    print("BERLIN:\n", seasonal_avg_b)
    print("\nMUNICH:\n", seasonal_avg_m)

    #mevsimsel ortalamayı karşılaştırır.
    print("\n--- Seasonal Pattern Comparison ---")
    b_windiest = seasonal_avg_b.idxmax()
    m_windiest = seasonal_avg_m.idxmax()
    print(f"Berlin's windiest season: {b_windiest} ({seasonal_avg_b.max():.2f} m/s)")
    print(f"Munich's windiest season: {m_windiest} ({seasonal_avg_m.max():.2f} m/s)")

    print("=" * 50)

    # YENİ EKLENEN KISIM: TASK 3 - İSTATİSTİKSEL ANALİZ
    print("\n" + "=" * 50)
    print("### STATISTICAL ANALYSIS (Extreme & Diurnal Patterns) ###")

    # Identify Days/Periods with Extreme Weather Conditions (Highest Wind Speeds)
    max_speed_b = df_b['wind_speed'].max()
    max_speed_m = df_m['wind_speed'].max()

    # Maksimum hızın kaydedildiği anı bulma
    #index, maximum anın indexini çeker.
    #.strftime indexi güne çevirir
    extreme_days_b = df_b[df_b['wind_speed'] == max_speed_b].index.strftime('%Y-%m-%d %H:%M')
    extreme_days_m = df_m[df_m['wind_speed'] == max_speed_m].index.strftime('%Y-%m-%d %H:%M')

    print("\n--- Extreme Wind Events (Max Recorded Speed) ---")
    print(f"BERLIN Max Speed: {max_speed_b:.2f} m/s, Occurred: {', '.join(extreme_days_b[:1])}")
    print(f"MUNICH Max Speed: {max_speed_m:.2f} m/s, Occurred: {', '.join(extreme_days_m[:1])}")

    # 2. Calculate Diurnal (Daily) Patterns in Wind Speed

    # df.index.hour kullanarak veriyi saatin 0'dan 23'e kadar olan değerlerine göre gruplama
    diurnal_pattern_b = df_b.groupby(df_b.index.hour)['wind_speed'].mean()
    diurnal_pattern_m = df_m.groupby(df_m.index.hour)['wind_speed'].mean()

    print("\n--- Diurnal (Daily Hour) Wind Speed Pattern (m/s) ---")
    print("Hour (0-23) vs. Average Wind Speed")

    # Günlük döngü ortalamasını bir tabloda gösterme
    df_diurnal = pd.DataFrame({'Berlin_Avg_Speed': diurnal_pattern_b,
        'Munich_Avg_Speed': diurnal_pattern_m})

    print(df_diurnal.to_string(float_format='%.3f'))

    # En rüzgarlı saati bulma
    b_peak_hour = diurnal_pattern_b.idxmax()
    m_peak_hour = diurnal_pattern_m.idxmax()

    print(f"\nBerlin Peak Hour: {b_peak_hour}:00 (Avg: {diurnal_pattern_b.max():.3f} m/s)")
    print(f"Munich Peak Hour: {m_peak_hour}:00 (Avg: {diurnal_pattern_m.max():.3f} m/s)")

    print("=" * 50)
    # Dönen değerleri, Task 3'ün çıktılarını da içerecek şekilde güncelliyoruz.
    return monthly_avg_b, monthly_avg_m, seasonal_avg_b, seasonal_avg_m, diurnal_pattern_b, diurnal_pattern_m


# ==============================================================================
# TASK 4: VISUALIZATION
# ==============================================================================

def plot_visualizations(monthly_avg_b, monthly_avg_m, seasonal_avg_b, seasonal_avg_m, diurnal_b, diurnal_m):
    print("\n" + "=" * 50)
    print("### 4. VISUALIZATION ###")

    # 1. Monthly Average Wind Speeds Time Series Plot (Çizgi Grafik)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Aylık ortalamalar zaten zaman indeksi olarak ayarlanmış Series nesneleridir.
    plt.plot(monthly_avg_b.index, monthly_avg_b.values, label='Berlin', marker='o')
    plt.plot(monthly_avg_m.index, monthly_avg_m.values, label='Munich', marker='o')

    plt.title('Monthly Average Wind Speed Comparison (2024)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Visualization 1: Monthly Average Time Series Plot generated.")

    # 2. Seasonal Comparison Bar Chart (Mevsimsel Karşılaştırma Çubuk Grafik)
    # --------------------------------------------------------------------------

    # Karşılaştırma için Seasonal ortalamaları tek bir DataFrame'de birleştirme
    df_seasonal = pd.DataFrame({
        'City': ['Berlin'] * 4 + ['Munich'] * 4,
        'Season': ['Winter', 'Spring', 'Summer', 'Autumn'] * 2,
        'Avg_Speed': seasonal_avg_b.tolist() + seasonal_avg_m.tolist()
    })

    plt.figure(figsize=(8, 6))
    # Seaborn kullanarak mevsimlere göre gruplanmış çubuk grafik oluşturma
    sns.barplot(
        x='Season',
        y='Avg_Speed',
        hue='City',
        data=df_seasonal,
        order=['Winter', 'Spring', 'Summer', 'Autumn'],
        palette='viridis'
    )

    plt.title('Seasonal Average Wind Speed Comparison', fontsize=16)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='City')
    plt.tight_layout()
    plt.show()

    print("Visualization 2: Seasonal Comparison Bar Chart generated.")

    # 3. Diurnal (Daily Hour) Pattern Plot (Günlük Örüntü Çizgi Grafik)
    # Bu, Wind Rose'a alternatif olarak, Task 3'teki verileri kullanır.
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Diurnal Series'lerin indeksi zaten 0'dan 23'e kadar saatleri içerir.
    plt.plot(diurnal_b.index, diurnal_b.values, label='Berlin', marker='.')
    plt.plot(diurnal_m.index, diurnal_m.values, label='Munich', marker='.')

    plt.title('Diurnal (Hourly) Wind Speed Pattern', fontsize=16)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.xticks(range(0, 24, 2))  # X eksenini 2'şer saatte bir etiketle
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Visualization 3: Diurnal Pattern Plot generated.")

    print("=" * 50)






if __name__ == "__main__":

    df_berlin_raw, df_munich_raw = load_and_explore_datasets(berlin_file, munich_file)

    if df_berlin_raw is not None and df_munich_raw is not None:
        print("\nSTEP 1 COMPLETE")

        # Task 2: Aylık Ortalamaları Hesaplama
        monthly_b, monthly_m, seasonal_b, seasonal_m,diurnal_b, diurnal_m = compute_monthly_averages(df_berlin_raw, df_munich_raw)
        print("\n[STEP 2 & 3 COMPLETE: TEMPORAL & STATISTICAL ANALYSIS]")

        plot_visualizations(monthly_b, monthly_m, seasonal_b, seasonal_m, diurnal_b, diurnal_m)

        print("\n[COMPLETED: VISUALIZATION]")


