#Yarışma Adresi: https://www.kaggle.com/competitions/datathon-2024
#Bu çalışmayı hazırlama niyetim kendimi denemek ve yeni bilgiler öğrenmektir.
#Bu sebeple bu çalışmada kendi fikir ve düşüncelerime ek olarak yarışma sonrasında paylaşılan notebooklardan da faydalandığımı belirtmek isterim.
import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)

# Specify the file path
train_path = r"C:\Users\Halil\PycharmProjects\BTK\BTK\datasets\train.csv"
test_path = r"C:\Users\Halil\PycharmProjects\BTK\BTK\datasets\test_x.csv"

# Read the CSV file into a DataFrame
train = pd.read_csv(train_path,low_memory=False)
test = pd.read_csv(test_path,low_memory=False)

#0-65124 train
df = pd.concat([train,test],axis=0,ignore_index=True)
df = df[df['Basvuru Yili'] != 2016]

df['Cinsiyet'] = df['Cinsiyet'].fillna("Belirtmek istemiyorum").replace({"Kadın":0,
                                                                         "Erkek":1,
                                                                         "ERKEK":1,
                                                                         "Belirtmek istemiyorum":2}).astype(float)

df['Universite Kacinci Sinif'] = df['Universite Kacinci Sinif'].replace({"Hazırlık":0,
                                                                         "hazırlık":0,
                                                                         "0":0,
                                                                         "1":1,
                                                                         "2":2,
                                                                         "3":3,
                                                                         "4":4,
                                                                         "5":5,
                                                                         "6":6,
                                                                         "Mezun":7,
                                                                         "Yüksek Lisans":8,
                                                                         "Tez":8}).astype(float)

df.loc[:, 'Universite Kacinci Sinif'] = df.groupby('Basvuru Yili')['Universite Kacinci Sinif'].transform(lambda x: x.fillna(x.median()))



df["Dogum Tarihi"] = pd.to_datetime(df["Dogum Tarihi"], errors="coerce")
df["Basvuru Yasi"] = df["Basvuru Yili"] - df["Dogum Tarihi"].dt.year

df['Basvuru Yasi'] = df['Basvuru Yasi'].fillna(18 + df['Universite Kacinci Sinif']).astype(float)
filtered_df = df[(df['Basvuru Yasi'] >= 18) & (df['Basvuru Yasi'] <= 26)]
df = filtered_df.reset_index(drop=True)

# Define city_organizer function
def city_organizer(city):
    if isinstance(city, str):
        city = unidecode(city).lower().strip()
        # Additional processing if needed
    else:
        city = "diger"
    return city

# Doğum yeri veya ikametgahdan birisi NaN ise diğeri ile dolduruldu!
df['Dogum Yeri'] = df['Dogum Yeri'].combine_first(df['Ikametgah Sehri'])
df['Ikametgah Sehri'] = df['Ikametgah Sehri'].combine_first(df['Dogum Yeri'])

df["Dogum Yeri"] = df["Dogum Yeri"].fillna("diger").apply(city_organizer)
df["Ikametgah Sehri"] = df["Ikametgah Sehri"].apply(city_organizer)
df["Lise Sehir"] = df["Lise Sehir"].fillna("diger").apply(city_organizer)

def rare_encoder(df, threshold=100):
    rare_columns = ['Dogum Yeri', 'Ikametgah Sehri', 'Lise Sehir']

    for var in rare_columns:
        # Her bir kategorinin frekansını alıyoruz
        tmp = df[var].value_counts()
        # Frekansı threshold'dan (100) az olanları 'diger' ile değiştireceğiz
        rare_labels = tmp[tmp < threshold].index
        # Rare olanları 'diger' olarak işaretliyoruz
        df[var] = np.where(df[var].isin(rare_labels), 'diger', df[var])

rare_encoder(df)


def university_name_organizer(row):
    # Harfleri küçük yapma, Türkçe karakterleri değiştirme
    if pd.notnull(row):
        row = unidecode(row).lower().strip()

        # Gereksiz kelimeleri temizleme (eğer gerekiyorsa)
        row = row.replace('universi̇tesi̇', 'universitesi')
        row = row.replace('univ', 'universitesi')
        row = row.replace('universitesiersitesi','universitesi')

        return row
    else:
        return row

df['Universite Adi'] = df['Universite Adi'].apply(university_name_organizer)
df["Universite Adi"].value_counts()
df["Universite Adi"].isnull().sum()

def rare_encoder2(df, rare_perc):

    tmp = df['Universite Adi'].value_counts() / len(df)
    rare_labels = tmp[tmp < rare_perc].index
    df['Universite Adi'] = np.where(df['Universite Adi'].isin(rare_labels), 'diger', df['Universite Adi'])

    return df

df = rare_encoder2(df, 0.005)
df['Universite Adi'] = df['Universite Adi'].fillna('diger')

uni_list={'beykent universitesi':'özel',
          'koc universitesi':'özel',
          'ozyegin universitesi':'özel',
          'bahcesehir universitesi':'özel',
          'istanbul aydin universitesi':'özel',
          'istanbul medipol universitesi':'özel',
          'yeditepe universitesi':'özel',
          'dokuz eylul universitesi':'devlet',
          'gazi universitesi':'devlet',
          'karabuk universitesi':'devlet',
          'firat universitesi':'devlet',
          'kocaeli universitesi':'devlet',
          'marmara universitesi':'devlet',
          'istanbul universitesi':'devlet',
          'ondokuz mayis universitesi':'devlet',
          'hacettepe universitesi':'devlet',
          'akdeniz universitesi':'devlet',
          'istanbul teknik universitesi':'devlet',
          'trakya universitesi':'devlet',
          'selcuk universitesi':'devlet',
          'bogazici universitesi':'devlet',
          'ege universitesi':'devlet',
          'yildiz teknik universitesi':'devlet',
          'anadolu universitesi':'devlet',
          'bursa uludag universitesi':'devlet',
          'mersin universitesi':'devlet',
          'eskisehir osmangazi universitesi':'devlet',
          'necmettin erbakan universitesi':'devlet',
          'orta dogu teknik universitesi':'devlet',
          'ataturk universitesi':'devlet',
          'afyon kocatepe universitesi':'devlet',
          'mugla sitki kocman universitesi':'devlet',
          'cukurova universitesi':'devlet',
          'sakarya universitesi':'devlet',
          'dicle universitesi':'devlet',
          'ankara universitesi':'devlet',
          'usak universitesi':'devlet',
          'erciyes universitesi':'devlet',
          'suleyman demirel universitesi':'devlet',
          'ihsan dogramaci bilkent universitesi':'özel',
          'sabanci universitesi':'özel'
          }

df.loc[df['Universite Turu'].isna(), 'Universite Turu'] = df['Universite Adi'].map(uni_list)

df['Universite Turu']=df['Universite Turu'].fillna('diger')
df['Universite Turu'] = df['Universite Turu'].str.lower().replace({"devlet":0,
                                                       "özel":1,
                                                       "diger":2}).astype(float)

def group_lise_bolum(value):
    value = str(value).lower()  # Convert to lowercase for easier matching

    if any(word in value for word in ["matematif","matematik","fen","mf","sayısalmf","bikimleri","M-F","sayısak","sayılsal","sayısal/fen"]):
        return 1
    elif any(word in value for word in ["elektirk","kalıp","radyo","makina","makine","büro","grafik","programcılıgı",
                                        "programcılığı","web","bilgisayar","bilişim","ulastirma","otomasyon","muhasebe",
                                        "güverte","elektrik","elektronik","kimya","cnc","bakım","onarım","matbaa","metal",
                                        "mekatronik","tekstil","uçak","yiyecek","ahsap","mobilya","harita","kadastro","yazilim",
                                        "yazılım","web","gıda","iklimlendirme","giyim","kontrol","endüstriyel","otomotiv","moda",
                                        "torna","tasarım","halkla","turizm","konaklama","gemi"]):
        return 2
    elif any(word in value for word in ["eşit","agırlık","edebiyat","sosyal","esitagirlik","ağirlik","eşitağirlik","türkçe-mat","türkçe-matematik"]):
        return 3
    elif any(word in value for word in ["TS","sözel","söz."]):
        return 4
    elif any(word in value for word in ["dil","yabancı","almanca","ydl","ingilizce"]):
        return 5
    else:
        return 0

df['Lise Bolumu'] = df['Lise Bolumu'].fillna(0).apply(group_lise_bolum).astype(float)

def group_lise_mezuniyet(value):
    value = str(value)
    if any(word in value for word in ["75 - 100","84-70","100-85","4.00-3.50","3.00-4.00","3.50-3.00","3.50-3"]):
        return 3
    elif any(word in value for word in ["50 - 75","69-55","3.00-2.50","50 - 74"]):
        return 2
    elif any(word in value for word in ["2.50 ve altı","54-45","25 - 50","44-0","0 - 25","25 - 49","0 - 24"]):
        return 1
    else:
        return 0

df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].fillna(0).apply(group_lise_mezuniyet)

df['Baska Bir Kurumdan Burs Aliyor mu?'] = df["Baska Bir Kurumdan Burs Aliyor mu?"].fillna("hayır").str.lower()
df['Baska Bir Kurumdan Burs Aliyor mu?']=df['Baska Bir Kurumdan Burs Aliyor mu?'].replace({"hayır":0,
                                                  "evet":1}).astype(float)

def burs_grupla(burs_adi):
    if pd.isna(burs_adi) or burs_adi.strip() == '':  # Boş ya da NaN ise boş döner
        return 0

    burs_adi = burs_adi.lower()  # Burs adını küçük harfe çeviriyoruz
    if re.search(r'\b(başarı|yks|ösym|tubitak|yök)\b', burs_adi):
        return 1
    elif re.search(r'\b(kredi|geri ödemeli)\b', burs_adi):
        return 2
    elif re.search(r'\b(yurtlar|kyk|başbakanlık|basbakanlık|gençlik ve spor|basbakanlık|t.c.|devlet|kyk-burs)\b', burs_adi):
        return 3
    elif re.search(r'\b(tev)\b', burs_adi):
        return 4
    else:
        return 0  # Eğer kategoriye uymuyorsa -Diger

df['Burs Aldigi Baska Kurum'] = df['Burs Aldigi Baska Kurum'].apply(burs_grupla)


def extract_numbers(text):
    if isinstance(text, str):
        numbers = re.findall(r'\d+', text)
        return int(max(numbers, key=int)) if numbers else 0
    return 0

def classify_miktar(miktar):
    if miktar == 0:
        return 0
    elif 0 < miktar < 250:
        return 1
    elif 250 <= miktar < 500:
        return 2
    elif 500 <= miktar < 750:
        return 3
    elif 750 <= miktar < 1000:
        return 4
    elif miktar >= 1000:
        return 5
    else:
        return 0

df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].apply(extract_numbers)
df['Baska Kurumdan Aldigi Burs Miktari'] = df['Baska Kurumdan Aldigi Burs Miktari'].apply(classify_miktar)

#DİPNOT: Bu değişkeni temize dökücek yeni bir fonksiyon, işlem üret.

df['Anne Egitim Durumu'] = df['Anne Egitim Durumu'].apply(
        lambda x: unidecode(x).lower() if isinstance(x, str) else x)

df["Anne Egitim Durumu"] = (df["Anne Egitim Durumu"].fillna(0).
                            replace({
                            "ilkokul mezunu": 1,
                            "lise":3,
                            "ilkokul":1,
                            "universite":4,
                            "egitim yok":0,
                            "ortaokul":2,
                            "yüksek lisans":5,
                            "doktora":5,
                            "lise mezunu": 3,
                            "universite mezunu": 4,
                            "egitimi yok": 0,
                            "ortaokul mezunu": 2,
                            "yuksek lisans / doktora": 5,
                            "yuksek lisans / doktara": 5,
                            "yuksek lisans":5})).astype(float)


df['Baba Egitim Durumu'] = df['Baba Egitim Durumu'].apply(
        lambda x: unidecode(x).lower() if isinstance(x, str) else x)
df["Baba Egitim Durumu"] = (df["Baba Egitim Durumu"].fillna(0).
                            replace({
                            "ilkokul mezunu": 1,
                            "ilkokul":1,
                            "lise mezunu":3,
                            "lise":3,
                            "universite mezunu":4,
                            "universite":4,
                            "yuksek lisans / doktara":5,
                            "doktora":5,
                            "ortaokul mezunu": 2,
                            "ortaokul":2,
                            "egitimi yok": 0,
                            "egitim yok":0,
                            "yuksek lisans / doktora": 5,
                            "0": 0,
                            "yuksek lisans":5})).astype(float)


df["Anne Calisma Durumu"] = df["Anne Calisma Durumu"].replace({
    "Hayır":0,
    "Evet":1,
    "Emekli":2,
    None:0}).astype(float)

df["Baba Calisma Durumu"] = df["Baba Calisma Durumu"].replace({
    "Hayır":0,
    "Evet":1,
    "Emekli":2,
    None:0}).astype(float)

df["Anne Sektor"] = df["Anne Sektor"].fillna("yok").apply(unidecode).str.lower()
df["Anne Sektor"] = df["Anne Sektor"].replace({"0": 0,
                                               "yok":0,
                                               "-": 0,
                                               "kamu":1,
                                               "ozel sektor":2,
                                               "diger":3}).astype(float)

df["Baba Sektor"] = df["Baba Sektor"].fillna("yok").apply(unidecode).str.lower()
df["Baba Sektor"] = df["Baba Sektor"].replace({"0": 0,
                                               "-": 0,
                                               "yok":0,
                                               "kamu":1,
                                               "ozel sektor":2,
                                               "diger":3}).astype(float)

df['Kardes Sayisi'] = df['Kardes Sayisi'].fillna(0).replace({'Kardeş Sayısı 1 Ek Bilgi Aile Hk. Anne Vefat': 1.}).astype(float)

df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] = \
    df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'].fillna("hayır").str.lower().replace(
        {
            "evet": 1.,
            "hayır": 0.
        }
    ).astype(float)

df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] = \
        df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'].fillna("hayır").str.lower().replace(
            {
                "evet": 1.,
                "hayır": 0.
            }
        ).astype(float)

df['Spor Dalindaki Rolunuz Nedir?'] = \
        df['Spor Dalindaki Rolunuz Nedir?'].fillna("yok").apply(unidecode).str.lower()
df['Spor Dalindaki Rolunuz Nedir?'].value_counts()

df['Spor Dalindaki Rolunuz Nedir?'] = df['Spor Dalindaki Rolunuz Nedir?'].replace({"-":0,
                                                                                   "yok":0,
                                                                                   "diger":1,
                                                                                   "takim oyuncusu":2,
                                                                                   "lider/kaptan":3,
                                                                                   "bireysel spor":4,
                                                                                   "bireysel":4,
                                                                                   "kaptan":3,
                                                                                   "kaptan / lider":3}).astype(float)

df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].fillna('Hayır').replace({"Hayır":0,
                                                                                                                           "Evet":1}).astype(float)

df["Aktif olarak bir STK üyesi misiniz?"] = df["Aktif olarak bir STK üyesi misiniz?"].fillna('Hayır').replace({"Hayır":0,
                                                                                                               "Evet":1}).astype(float)

df["Ingilizce Biliyor musunuz?"] = df["Ingilizce Biliyor musunuz?"].fillna('Hayır').replace({'Hayır':0,
                                                                                             'Evet':1}).astype(float)

# Kelime sayısına göre metin uzunluğunu kategorilere ayıran fonksiyon
def categorize_word_count(column):
    return pd.cut(column.str.split().str.len(),
                  bins=[-np.inf, 5, 10, np.inf],
                  labels=[0, 1, 2]).astype("float")

df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"] = df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"].fillna("")
df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"] = categorize_word_count(
        df["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"])


def categorize_bolum(bolum):
    # Ensure the value is a string, otherwise return 'Diğer'
    if isinstance(bolum, str):
        bolum = bolum.lower()

        # Engineering
        if any(word in bolum for word in ['mühendis', 'mühendisligi', 'mühendisliği','engineering','muhendisligi','muhendisliği','mekatronik']):
            return 1

        # Social Sciences
        elif any(word in bolum for word in ['hukuk', 'işletme', 'iktisat', 'psikoloji', 'sosyoloji', 'sosyal',
                                            'ilişkiler', 'kamu', 'siyaset', 'finans', 'yönetim', 'maliye']):
            return 2

        # Medical & Health
        elif any(word in bolum for word in ['tıp', 'hemşire', 'eczacılık', 'diş hekimliği', 'sağlık',
                                            'fizyoterapi', 'hekim']):
            return 3

        # Natural Sciences
        elif any(word in bolum for word in ['matematik', 'fizik', 'kimya', 'biyoloji', 'fen']):
            return 4

        # Education
        elif 'öğretmen' in bolum or 'eğitim' in bolum:
            return 5

        # Art & Design
        elif any(word in bolum for word in ['güzel sanatlar', 'mimarlık', 'tasarım', 'sinema']):
            return 6

        # Other
        else:
            return 0
    else:
        # If it's not a string, return 'Diğer'
        return 0

df["Bölüm"] = df["Bölüm"].apply(categorize_bolum)

df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].fillna(0).replace({"3.00-2.50":2,
                                                                                     "2.50 ve altı":2,
                                                                                     "3.00 - 3.50":3,
                                                                                     "3.50-3":3,
                                                                                     "2.50 - 2.99":2,
                                                                                     "2.50 - 3.00":2,
                                                                                     "3.00 - 3.49":3,
                                                                                     "3.50 - 4.00":4,
                                                                                     "1.80 - 2.49":2,
                                                                                     "2.00 - 2.50":2,
                                                                                     "ORTALAMA BULUNMUYOR":5,
                                                                                     "Hazırlığım":5,
                                                                                     "2.50 -3.00":3,
                                                                                     "3.00 - 4.00":4,
                                                                                     "Not ortalaması yok":5,
                                                                                     "4-3.5":4,
                                                                                     "0 - 1.79":1,
                                                                                     "Ortalama bulunmuyor":5,
                                                                                     "1.00 - 2.50":1,
                                                                                     "4.0-3.5":4}).astype(float)


df["Burs Aliyor mu?"] = df["Burs Aliyor mu?"].fillna(0).str.lower().replace({"hayır":0,
                                                                   "evet":1}).astype(float)


df["Lise Turu"] = df["Lise Turu"].str.lower()
df["Lise Turu"] = df["Lise Turu"].fillna(0).replace({"anadolu lisesi":1,
                                                     "devlet":1,
                                                     "diğer":0,
                                                     "düz lise":1,
                                                     "özel":2,
                                                     "meslek lisesi":1,
                                                     "fen lisesi":1,
                                                     "meslek":1,
                                                     "özel lisesi":2,
                                                     "i̇mam hatip lisesi":1,
                                                     "özel lise":2}).astype(float)


def group_lise_ad(value):
    value = unidecode(str(value)).lower()

    if any(word in value for word in ["anadolu","cok","programlı","a.l","düz"]):
        return 1
    elif any(word in value for word in ["meslek","saglık","teknik","mtal","m.t.a.l"]):
        return 2
    elif any(word in value for word in ["fen","sosyal"]):
        return 3
    elif any(word in value for word in ["imam","hatip","aihl","a.i.h.l","i.h.l","ihl"]):
        return 4
    elif any(word in value for word in ["acıkögretim"]):
        return 5
    elif any(word in value for word in ["kolej","özel","koleji"]):
        return 6
    else:
        return 0


df['Lise Adi'] = df['Lise Adi'].fillna(0).apply(group_lise_ad).astype(float)

dropcols = [
    "Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite",
    "Ingilizce Seviyeniz?",
    "Stk Projesine Katildiniz Mi?",
    "Hangi STK'nin Uyesisiniz?",
    "Uye Oldugunuz Kulubun Ismi",
    "Lise Bolum Diger",
    "Lise Adi Diger",
    "Daha Once Baska Bir Universiteden Mezun Olmus",
    "Burslu ise Burs Yuzdesi",
    "Dogum Tarihi"
]

def drop_columns(dataframe, columns):
    return dataframe.drop(columns=columns)

df = drop_columns(df,dropcols)


university_city_map = {
    'istanbul universitesi': 'istanbul',
    'marmara universitesi': 'istanbul',
    'istanbul teknik universitesi': 'istanbul',
    'yildiz teknik universitesi': 'istanbul',
    'orta dogu teknik universitesi': 'ankara',
    'dokuz eylul universitesi': 'izmir',
    'bogazici universitesi': 'istanbul',
    'kocaeli universitesi': 'kocaeli',
    'hacettepe universitesi': 'ankara',
    'gazi universitesi': 'ankara',
    'selcuk universitesi': 'konya',
    'ege universitesi': 'izmir',
    'ankara universitesi': 'ankara',
    'anadolu universitesi': 'eskisehir',
    'sakarya universitesi': 'sakarya',
    'akdeniz universitesi': 'antalya',
    'erciyes universitesi': 'kayseri',
    'suleyman demirel universitesi': 'isparta',
    'karadeniz teknik universitesi': 'trabzon',
    'cukurova universitesi': 'adana',
    'karabuk universitesi': 'karabuk',
    'ataturk universitesi': 'erzurum',
    'firat universitesi': 'elazig',
    'gaziantep universitesi': 'gaziantep',
    'pamukkale universitesi': 'denizli',
    'uludag universitesi': 'bursa',
    'eskisehir osmangazi universitesi': 'eskisehir',
    'trakya universitesi': 'edirne',
    'koc universitesi': 'istanbul',
    'ondokuz mayis universitesi': 'samsun',
    'mugla sitki kocman universitesi': 'mugla',
    'dicle universitesi': 'diyarbakir',
    'mersin universitesi': 'mersin',
    'inonu universitesi': 'malatya',
    'bahcesehir universitesi': 'istanbul',
    'ozyegin universitesi': 'istanbul',
    'yeditepe universitesi': 'istanbul',
    'necmettin erbakan universitesi': 'konya',
    'beykent universitesi': 'istanbul',
    'balikesir universitesi': 'balikesir',
    'istanbul medipol universitesi': 'istanbul',
    'duzce universitesi': 'duzce',
    'bursa uludag universitesi': 'bursa',
    'usak universitesi': 'usak',
    'celal bayar universitesi': 'manisa',
    'afyon kocatepe universitesi': 'afyon',
    'istanbul bilgi universitesi': 'istanbul',
    'istanbul aydin universitesi': 'istanbul',
    'izmir katip celebi universitesi': 'izmir',
    'kirikkale universitesi': 'kirikkale',
    'diger': 'diger',
    'ihsan dogramaci bilkent universitesi': 'ankara',
    'sabanci universitesi': 'istanbul'
}

# Map the 'Universite Adi' column to the corresponding city using the dictionary
df['Universite Sehri'] = df['Universite Adi'].map(university_city_map)
df['Sehir Farki'] = (df['Ikametgah Sehri'] != df['Universite Sehri']).astype(int)

# Üniversite Kaçıncı Sınıf 4'ten büyükse, 4 olarak ayarlama
df['Universite Sinif Hesap'] = df['Universite Kacinci Sinif'].apply(lambda x: 4 if x > 4 else x)

# Üniversite Not Ortalaması ve ayarlanmış Üniversite Kaçıncı Sınıf'ı çarpma
uni_component = df['Universite Not Ortalamasi'] * df['Universite Sinif Hesap']

# Akademik skor hesaplama: %40 Lise Mezuniyet Notu + %60 Üniversite bileşeni
df['Akademik Skor'] = (0.4 * df['Lise Mezuniyet Notu']) + (0.6 * uni_component)

df['Aile Egitim'] = df['Anne Egitim Durumu'].astype(str) + '_' + df['Baba Egitim Durumu'].astype(str)

df['Anne Bilgi'] = df['Anne Calisma Durumu'].astype(str) + '_' + df['Anne Sektor'].astype(str)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
target_correlation = numeric_df.corr()['Degerlendirme Puani'].sort_values(ascending=False)


# Korelasyon değerleriyle ağırlıkları belirleyelim
weights = {
    'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?': 0.592171,
    'Aktif olarak bir STK üyesi misiniz?': 0.453261,
    'Spor Dalindaki Rolunuz Nedir?': 0.329681,
    'Profesyonel Bir Spor Daliyla Mesgul musunuz?': 0.345793,
    'Girisimcilikle Ilgili Deneyiminiz Var Mi?': 0.363545,
    'Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?': 0.218204,
}

# Ağırlıkların toplamı ile normalize et
total_weight = sum(weights.values())
normalized_weights = {k: v / total_weight for k, v in weights.items()}

# Sosyalite Skor'u ağırlıklarla hesapla
df['Sosyalite Skor'] = (
    df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] * normalized_weights['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] +
    df['Aktif olarak bir STK üyesi misiniz?'] * normalized_weights['Aktif olarak bir STK üyesi misiniz?'] +
    df['Spor Dalindaki Rolunuz Nedir?'] * normalized_weights['Spor Dalindaki Rolunuz Nedir?'] +
    df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] * normalized_weights['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] +
    df['Girisimcilikle Ilgili Deneyiminiz Var Mi?'] * normalized_weights['Girisimcilikle Ilgili Deneyiminiz Var Mi?'] +
    df['Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?'] * normalized_weights['Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?']
)

correlation2 = df['Akademik Skor'].corr(df['Degerlendirme Puani'])
correlation3 = df['Sosyalite Skor'].corr(df['Degerlendirme Puani'])


# Ağırlıkları hesaplamak için korelasyon değerlerini normalize et
total_correlation = correlation3 + correlation2
weight_social = correlation3 / total_correlation
weight_academic = correlation2 / total_correlation

# Yeni skoru ağırlıklarla hesapla
df['Genel Skor'] = (df['Sosyalite Skor'] * weight_social) + (df['Akademik Skor'] * weight_academic)

df.drop(columns=['Universite Sinif Hesap'], inplace=True)


# 'Basvuru Yili' 2023 olanları filtreleyip yeni bir DataFrame'e kaydedelim
df_2023 = df[df['Basvuru Yili'] == 2023]

# 'Basvuru Yili' 2023 olmayanları filtreleyip yeni bir DataFrame'e kaydedelim
df_train = df[df['Basvuru Yili'] != 2023]

df_train['Degerlendirme Puani'].isnull().sum()
df_train['Degerlendirme Puani'] = df_train['Degerlendirme Puani'].fillna(0)

df_2023['Basvuru Yili'].value_counts()
df_2023['Degerlendirme Puani'].isnull().sum()


# Hedef değişken ve özellikleri ayır
target = 'Degerlendirme Puani'
X = df_train.drop(columns=[target])
y = df_train[target]

# Kategorik sütunlar
one_hot_columns = ['Anne Bilgi']
target_encoding_columns = ['Dogum Yeri', 'Ikametgah Sehri', 'Universite Adi',
                           'Universite Sehri', 'Aile Egitim', 'Lise Sehir']

# 2. One-Hot Encoding
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
one_hot_encoded = pd.DataFrame(one_hot_encoder.fit_transform(X[one_hot_columns]),
                               columns=one_hot_encoder.get_feature_names_out(one_hot_columns),
                               index=X.index)

# 3. Target Encoding
target_encoder = ce.TargetEncoder(cols=target_encoding_columns)
target_encoded = target_encoder.fit_transform(X[target_encoding_columns], y)

# 4. Encode edilmiş veriyi birleştir
X_transformed = pd.concat([X.drop(columns=one_hot_columns + target_encoding_columns),
                           one_hot_encoded, target_encoded], axis=1)

# 5. Eğitim ve test verisini ayır
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


# 6. XGBoost modelini hiperparametrelerle eğit
xgb_model = XGBRegressor(
    colsample_bytree=0.36126005385284843,
    learning_rate=0.09966376517945073,
    max_depth=9,
    n_estimators=167,
    subsample=1.0,
    reg_lambda=0.10973595955953998,  # L2 regularization
    reg_alpha=0.02992072370774055,    # L1 regularization
    min_child_weight=75,
    colsample_bynode=0.6736583774979139,
    random_state=42
)

# Modeli eğit
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate the model's performance
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)


# 3. Encode işlemleri
# One-Hot Encoding
one_hot_encoded = pd.DataFrame(one_hot_encoder.transform(df_2023[['Anne Bilgi']]),
                               columns=one_hot_encoder.get_feature_names_out(['Anne Bilgi']),
                               index=df_2023.index)

# Target Encoding
target_encoded = target_encoder.transform(df_2023[['Dogum Yeri', 'Ikametgah Sehri', 'Universite Adi',
                                                     'Universite Sehri', 'Aile Egitim', 'Lise Sehir']])

# Encode edilmiş veriyi birleştir
test_data_transformed = pd.concat([df_2023.drop(columns=['Anne Bilgi', 'Dogum Yeri', 'Ikametgah Sehri',
                                                           'Universite Adi', 'Universite Sehri', 'Aile Egitim',
                                                           'Lise Sehir']),
                                   one_hot_encoded, target_encoded], axis=1)

# Assuming test_data_transformed includes 'Degerlendirme Puani', drop it
if 'Degerlendirme Puani' in test_data_transformed.columns:
    test_data_transformed = test_data_transformed.drop('Degerlendirme Puani', axis=1)


# 4. Tahmin yapma
test_predictions = xgb_model.predict(test_data_transformed)

# 5. Tahmin sonuçlarını test verisine ekleme
df_2023['Degerlendirme Puani'] = test_predictions

result = df_2023[['id', 'Degerlendirme Puani']]

# Sonucu CSV dosyasına kaydet
result.to_csv('datasets\main_predictions.csv', index=False)

print("Tahminler başarıyla kaydedildi ve main_predictions.csv dosyasına kaydedildi.")
