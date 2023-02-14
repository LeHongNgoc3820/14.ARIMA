# ARIMA MODEL

[**Chi tiết bài viết**](https://github.com/LeHongNgoc3820/14.ARIMA/blob/main/Forecasting%20Air%20Passengers%20with%20AIRIMA.ipynb)

**Dưới đây là nội dung tóm tắt**

## AN INTRODUCTION TO ARIMA MODEL THEORY

<a class="anchor" id="0.1"></a>
### Table of Contents
+ [**1. Giới thiệu về Time Series**](#1)
    + [1.1 Khái niệm Time Series](#1.1)
    + [1.2 Các thành phần trong Time Series](#1.2)
    + [1.3 Ứng dụng](#1.3)
    + [1.4 Stationarity Tests in Time Series](#1.4)
+ [**2. Mô hình ARIMA**](#2)
    + [2.1 Giới thiệu mô hình ARIMA](#2.1)
    + [2.2 Các thành phần trong ARIMA model](#2.2)
    + [2.3 Công thức ARIMA model](#2.3)
    + [2.4 Lựa chọn tham số ARIMA (p, d, q)](#2.4)
    + [2.5 Giới thiệu phương pháp auto ARIMA](#2.5)
    + [2.6 Mô hình ARIMA mùa vụ (Seasonal ARIMA - SARIMA)](#2.6)
        + [2.6.1 Kiểm tra yếu tố mùa vụ](#2.6.1)
        + [2.6.2 Công thức](#2.6.2)
        + [2.6.3 Xây dựng mô hình SARIMA](#2.6.3)
    + [2.7 Mô hình ARIMAX và SARIMAX](#2.7)
        + [2.7.1 Giới thiệu](#2.7.1)
        + [2.7.2 Công thức](#2.7.2)
        + [2.7.3 Xây dựng mô hình ARIMAX và SARIMAX](#2.7.3)
    + [2.8 Xác định độ chính xác của mô hình](#2.8)
+ [**3. Ưu và khuyết điểm của mô hình ARIMA**](#3)
    + [3.1 Ưu điểm](#3.1)
    + [3.2 Khuyết điểm](#3.2)
+ [**4. Kết luận**](#4)
+ [**5. Câu hỏi ôn tập**](#5)

## 1. Giới thiệu về Time Series <a class="anchor" id="1"></a>
[**Table of Contents**](#0.1)

### 1.1 Khái niệm Time Series <a class="anchor" id="1.1"></a>
Chuỗi thời gian (time series) là một chuỗi các điểm dữ liệu xảy ra theo thứ tự liên tiếp trong một khoảng thời gian. Một chuỗi thời gian sẽ theo dõi chuyển động của các điểm dữ liệu đã chọn (chẳng hạn như giá của chứng khoán) trong một khoảng thời gian xác định.

Ứng dụng của chuỗi thời gian trải khắp các ngành công nghiệp khác nhau như: quan sát hoạt động sóng điện trong não, đo lượng mưa, dự báo giá cổ phiếu, theo dõi doanh số bán lẻ hàng năm, người đăng ký hàng tháng, nhịp tim mỗi phút,...

Dữ liệu chuỗi thời gian là tập hợp các quan sát thu được thông qua các phép đo lặp lại theo thời gian. Dữ liệu chuỗi thời gian ở khắp mọi nơi, vì thời gian là thành phần của mọi thứ mà chúng ta có thể nhận biết được.

Dự báo chuỗi thời gian là một lớp mô hình quan trọng trong thống kê, kinh tế lượng và machine learning. Sở dĩ chúng ta gọi lớp mô hình này là chuỗi thời gian (time series) là vì mô hình được áp dụng trên các chuỗi đặc thù có yếu tố thời gian. Một mô hình chuỗi thời gian thường dự báo dựa trên giả định rằng các qui luật trong quá khứ sẽ lặp lại ở tương lai. Do đó xây dựng mô hình chuỗi thời gian là chúng ta đang mô hình hóa mối quan hệ trong quá khứ giữa biến độc lập (biến đầu vào) và biến phụ thuộc (biến mục tiêu). Dựa vào mối quan hệ này để dự đoán giá trị trong tương lai của biến phụ thuộc.

Do là dữ liệu chịu ảnh hưởng bởi tính chất thời gian nên chuỗi thời gian thường xuất hiện những qui luật đặc trưng như : yếu tố chu kỳ, mùa vụ và yếu tố xu hướng. Đây là những đặc trưng thường thấy và xuất hiện ở hầu hết các chuỗi thời gian.

### 1.2 Các thành phần trong Time Series <a class="anchor" id="1.2"></a>
**Các thành phần trong Time Series:**

**Trend**
+ Hướng chung của dữ liệu theo thời gian. Tăng hoặc giảm dài hạn trong dữ liệu. Có thể được xem như là một độ dốc - slope (không phải là tuyến tính) gần như đi xuyên qua dữ liệu. Ví dụ, nếu chúng ta đang xem xét chiều cao của một đứa trẻ sơ sinh, chiều cao của chúng sẽ theo xu hướng tăng lên khi chúng còn nhỏ. Mặt khác, một người nào đó trong chương trình giảm cân thành công sẽ thấy cân nặng của họ có xu hướng giảm dần theo thời gian.

**Seasonality + Cycles**
+ Bất kỳ mẫu theo mùa hoặc lặp lại với tần suất cố định. Có thể là hàng giờ, hàng tháng, hàng ngày, hàng năm, v.v. Một chuỗi thời gian được cho là thời vụ khi nó bị ảnh hưởng bởi các yếu tố theo mùa (giờ trong ngày, tuần, tháng, năm, ...). Tính thời vụ có thể được quan sát với các mẫu chu kỳ (cyclical patterns) có tần số cố định (fixed frequency). Một ví dụ về điều này là doanh số bán áo khoác mùa đông tăng trong những tháng mùa đông và giảm trong những tháng mùa hè. Một ví dụ khác về điều này có thể là số dư tài khoản ngân hàng của bạn. Trong 10 ngày đầu mỗi tháng, số dư của bạn có xu hướng giảm khi bạn trả tiền thuê hàng tháng, tiền điện nước và các khoản thanh toán hóa đơn khác.
+ Một chu kỳ xảy ra khi dữ liệu biểu hiện tăng và giảm không có tần số cố định. Những biến động này thường là do điều kiện kinh tế và thường liên quan đến "Business cycle". Thời gian của những biến động này thường ít nhất là 2 năm.

**Irregularities + Noise**
+ Bất kỳ điểm đột biến hoặc điểm lõm lớn nào trong dữ liệu. Một ví dụ về điều này có thể là nhịp tim của bạn khi bạn chạy đường chạy 400 mét. Khi bạn bắt đầu cuộc đua, nhịp tim của bạn tương tự như nhịp tim trong suốt cả ngày, nhưng trong suốt cuộc đua, nó tăng vọt lên mức cao hơn nhiều trong một khoảng thời gian ngắn trước khi trở lại mức bình thường.

**Residuals**: Mỗi chuỗi thời gian có thể được phân tách thành hai phần:
+ **Forecast**: bao gồm một hoặc một số giá trị dự báo (forecasted values)
+ **Residuals**: sự khác biệt giữa một quan sát (observation) và giá trị được dự đoán của nó ở mỗi time step.
    
    $$ \text{Value of series at time t = (Predicted value at time t) + (Residual at time t)} $$

### 1.3 Ứng dụng <a class="anchor" id="1.3"></a>
+ Phân tích chuỗi thời gian (Time series analysis) có thể được sử dụng trong vô số ứng dụng kinh doanh để dự báo số lượng trong tương lai và giải thích các mô hình lịch sử của nó. Ví dụ các trường hợp ứng dụng:
    + Giải thích các mô hình theo mùa trong bán hàng.
    + Dự đoán số lượng khách hàng đến hoặc đi dự kiến.
    + Ước tính ảnh hưởng của một sản phẩm mới ra mắt về số lượng bán.
    + Phát hiện các sự kiện bất thường và ước tính mức độ ảnh hưởng của chúng.
    
### 1.4 Stationarity Tests in Time Series <a class="anchor" id="1.4"></a>
### ADF test
ADF test is used to determine the presence of unit root in the series, and hence helps in understanding if the series is stationary or not. The null and alternate hypothesis of this test are:
+ Null Hypothesis: The series has a unit root, meaning it is non-stationary. It has some time dependent structure.
+ Alternate Hypothesis: The series has no unit root, meaning it is stationary. It does not have time-dependent structure.

If the null hypothesis failed to be rejected, this test may provide evidence that the series is non-stationary.

A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).

### KPSS test
The null and alternate hypothesis for the KPSS test is opposite that of the ADF test.
+ Null Hypothesis: The process is trend stationary.
+ Alternate Hypothesis: The series has a unit root (series is not stationary).

A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (non-stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (stationary).

**When applying those tests the following outcomes are possible:**

Case 1: Both tests conclude that the series is not stationary - The series is not stationary

Case 2: Both tests conclude that the series is stationary - The series is stationary

Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.

Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

## 2. Mô hình ARIMA <a class="anchor" id="2"></a>
[**Table of Contents**](#0.1)

### 2.1 Giới thiệu mô hình ARIMA <a class="anchor" id="2.1"></a>
ARIMA (Auto Regressive Integrated Moving Average) là một lớp mô hình dự đoán phổ biến và linh hoạt sử dụng thông tin lịch sử để đưa ra dự đoán. Loại mô hình này là một kỹ thuật dự đoán cơ bản có thể được sử dụng làm nền tảng cho các mô hình phức tạp hơn.

**Types of ARIMA Model**:
+ **ARIMA**: Non-seasonal Autoregressive Integrated Moving Averages
+ **SARIMA**: Seasonal ARIMA
+ **SARIMAX**: Seasonal ARIMA with exogenous variables

### Ghi chú:
+ ARIMA được ứng dụng thường xuyên cho các dãy dữ liệu theo thời gian ổn định (Stationary time series).
+ Trong thống kê, dữ liệu thời gian ổn định là dữ liệu mà các chỉ số thống kế không đổi (trung bình, phương sai, hệ số tương quan, ...) theo thời gian.
+ Khi trung bình và phương sai có xu hướng biến chuyển theo thời gian thì sẽ có dữ liệu bất ổn định (non-stationary time series). Lúc này chúng ta phải tính bậc (order) của `d` để có được dữ liệu ổn định.
+ Nếu mô hình có thành phần theo mùa, chúng ta sử dụng mô hình ARIMA theo mùa (SARIMA). Trong trường hợp đó, sẽ có một bộ tham số khác: P, D và Q mô tả các liên kết tương tự như p, d và q nhưng tương ứng với các thành phần theo mùa của mô hình (Seasonal Model).

### 2.2 Các thành phần trong ARIMA model <a class="anchor" id="2.2"></a>
Mô hình ARIMA là viết tắt của "Auto-Regressive Integrated Moving Average" và có thể được chia thành **AR, I, MA**.

+ **AR(p) Autoregression** – a regression model that utilizes the dependent relationship between a current observation and observations over a previous period.An auto regressive (AR(p)) component refers to the use of past values in the regression equation for the time series.
+ **I(d) Integration** – uses differencing of observations (subtracting an observation from observation at the previous time step) in order to make the time series stationary. Differencing involves the subtraction of the current values of a series with its previous values d number of times.
+ **MA(q) Moving Average** – a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. A moving average component depicts the error of the model as a combination of previous error terms. The order q represents the number of terms to be included in the model.

+ Trong mô hình ARIMA có 3 tham số được sử dụng để giúp mô hình hoá các khía cạnh chính của một chuỗi thời gian: seasonality, trend và noise. Các tham số này được gắn nhãn lần lượt là p, d và q.
+ Một mô hình ARIMA thường được ghi là `ARIMA(p,d,q)`.
+ **Trong đó:**
    + `p`: là tham số kết hợp với khía cạnh tự động hồi quy của mô hình (auto-regressive aspect - AR), kết hợp các giá trị trong quá khứ mang tính chất lâu dài (giá trị quan sát hiện tại phụ thuộc vào các giá trị trước đó). (Tổng trọng số của các giá trị độ trễ của series). **Ví dụ**: Dự báo rằng nếu trời mưa nhiều trong vài ngày qua, có thể cho biết ngày mai trời sẽ mưa.
    + `d` (difference): là tham số kết hợp với phần tích hợp của mô hình (integrated part - I), nó ảnh hưởng đến lượng chênh lệch áp dụng cho một chuỗi thời gian (Sự khác biệt của time series). **Ví dụ**: Dự báo rằng lượng mưa ngày mai sẽ tương tự như lượng mưa ngày hôm nay, nếu lượng mưa hàng ngày tương tự trong vài ngày qua.
    + `q`: là tham số liên quan đến phần trung bình động của mô hình (moving average part - MA, các số liệu phụ thuộc nhau trong một khoảng thời gian ngắn). (Tổng số các lỗi dự báo bị trễ của series)

### a. Autoregressive Component - AR(p)
Thành phần tự hồi quy (autoregressive component) của mô hình ARIMA được đại diện bởi AR(p), với tham số `p` xác định số chuỗi bị trễ mà chúng tôi sử dụng.

$$AR(p) = \phi_0 + {\phi_1}{x_{t-1}} + {\phi_2}{x_{t-1}} + ... + {\phi_p}{x_{t-p}}$$

**AR(0): White Noise**
+ Nếu chúng ta đặt tham số `p` là 0 (AR(0)), không có số hạng tự hồi quy. Chuỗi thời gian này chỉ là White Noise.
+ Mỗi điểm dữ liệu được lấy mẫu từ một phân phối có giá trị trung bình bằng 0 và phương sai của bình phương sigma.
+ Điều này dẫn đến một dãy số ngẫu nhiên không thể dự đoán được.
+ Điều này thực sự hữu ích vì nó có thể đóng vai trò là một giả thuyết vô hiệu và bảo vệ các phân tích của chúng tôi khỏi việc chấp nhận các mẫu dương tính giả (false-positive).

**AR(1): Random Walks and Oscillations**
+ Với tham số p được đặt thành 1, chúng tôi đang tính đến dấu thời gian (timestamps) trước đó được điều chỉnh theo hệ số nhân, sau đó thêm nhiễu trắng.
+ Nếu hệ số nhân bằng 0 thì chúng ta nhận được White Noise và nếu hệ số nhân là 1 thì chúng ta sẽ có bước đi ngẫu nhiên (Random Walks).
+ Nếu hệ số nhân nằm trong khoảng từ $0 < {\alpha_1} < 1$, thì chuỗi thời gian sẽ thể hiện độ đảo ngược trung bình. Điều này có nghĩa là các giá trị có xu hướng xoay quanh 0 và trở lại giá trị trung bình sau khi hồi quy từ nó.

**AR(p): Higher-order terms**
+ Tăng tham số p hơn nữa chỉ có nghĩa là quay trở lại xa hơn và thêm nhiều dấu thời gian (timestamps) hơn được điều chỉnh bởi hệ số riêng của chúng.
+ Chúng ta có thể quay lại bao xa tùy ý, nhưng khi quay lại xa hơn, có nhiều khả năng chúng ta nên sử dụng các tham số bổ sung như đường trung bình động (MA(q)).

### b. Moving Average - MA(q)
"This component is not a rolling average, but rather the lags in the white noise."

$$\text{MA(q)} = \mu + \sum_{i=1}^q{\theta_i}{\epsilon_{t-i}} $$

**MA(q)**
+ MA(q) là mô hình trung bình động và q là số thuật ngữ lỗi dự báo bị trễ trong dự đoán.
+ Trong mô hình MA(1), dự báo của chúng tôi là một số hạng không đổi cộng với số hạng nhiễu trắng trước đó nhân với một số nhân, được cộng với thuật ngữ nhiễu trắng hiện tại.
+ Đây chỉ là xác suất + thống kê đơn giản vì chúng tôi đang điều chỉnh dự báo của mình dựa trên các thuật ngữ white noise trước đó.

### c. Intergrated (d)
Intergrated là quá trình đồng tích hợp hoặc lấy sai phân. Yêu cầu chung của các thuật toán trong time series là chuỗi phải đảm bảo tính dừng. Hầu hết các chuỗi đều tăng hoặc giảm theo thời gian. Do đó yếu tố tương quan giữa chúng chưa chắc là thực sự mà là do chúng cùng tương quan theo thời gian. Khi biến đổi sang chuỗi dừng, các nhân tố ảnh hưởng thời gian được loại bỏ và chuỗi sẽ dễ dự báo hơn. Để tạo thành chuỗi dừng, một phương pháp đơn giản nhất là chúng ta sẽ lấy sai phân. Một số chuỗi tài chính còn qui đổi sang logarit hoặc lợi suất. Bậc của sai phân để tạo thành chuỗi dừng còn gọi là bậc của quá trình đồng tích hợp (order of intergration). Quá trình của chuỗi dữ liệu được thực hiện như sau:

$$ \text{Sai phân bậc 1: I(1)} = \Delta{x_t} = x_t - x_{t-1} $$

Thông thường chuỗi sẽ dừng sau quá trình đồng tích hợp I(0) hoặc I(1). Rất ít chuỗi chúng ta phải lấy tới sai phân bậc 2. Một số trường hợp chúng ta cần biến đổi logarit hoặc căn bậc 2 để tạo thành chuỗi dừng.

### 2.3 Công thức ARIMA model <a class="anchor" id="2.3"></a>
Phương trình hồi quy ARIMA (p, d, q) có thể được biểu diễn dưới dạng công thức toán như sau:

$$\Delta{x_t} = {\phi_1}{\Delta{t-1}} + {\phi_2}{\Delta{t-2}} + {\phi_p}{\Delta{t-p}} + {\theta_1}{\epsilon_{t-1}} + {\theta_2}{\epsilon_{t-2}} + ... + {\theta_q}{\epsilon_{t-q}} $$

Trong đó: $\Delta{x_t}$ là giá trị sai phân bậc `d` và ${\epsilon_t}$ là các chuỗi nhiễu trắng.

Như vậy về tổng quát thì ARIMA là mô hình kết hợp của 2 quá trình tự hồi qui và trung bình trượt. Dữ liệu trong quá khứ sẽ được sử dụng để dự báo dữ liệu trong tương lai. Trước khi huấn luyện mô hình, cần chuyển hóa chuỗi sang chuỗi dừng bằng cách lấy sai phân bậc 1 hoặc logarit. Ngoài ra mô hình cũng cần tuân thủ điều kiện ngặt về sai số không có hiện tượng tự tương quan và phần dư là nhiễu trắng. Đó là lý thuyết của kinh tế lượng. Còn theo trường phái machine learning thì tôi chỉ cần quan tâm đến làm sao để lựa chọn một mô hình có sai số dự báo là nhỏ nhất.

### 2.4 Lựa chọn tham số ARIMA (p, d, q) <a class="anchor" id="2.4"></a>

+ `d`: d sẽ là bậc dừng của chuỗi dữ liệu.
+ `p`: Có thể dùng **Partial Autocorrelation (PACF)** plot để xác định.
+ `q`: Có thể dùng **Autocorrelation (ACF)** plot để xác định.

Trong python, có thể trình bày nhanh cả 2 đồ thị ACF và PACF bằng cách dùng package `plot_acf`, `plot_pacf`. Để sử dụng được các package đó, ta cần import vào thư viện như sau:`from statsmodels.graphics.tsaplots import plot_acf, plot_pacf`.

### 2.5 Giới thiệu phương pháp auto ARIMA <a class="anchor" id="2.5"></a>
Chúng ta thấy rằng việc lựa chọn mô hình tốt nhất chỉ đơn thuần dựa trên chỉ số AIC, khá đơn giản. Do đó, chúng ta hoàn toàn có thể tự động thực hiện quy trình này. Trên Python đã hỗ trợ tìm kiếm mô hình ARIMA phù hợp thông qua package `auto arima`. Chúng hoạt động như một grid search mà tham số chúng ta truyền vào chỉ là có hệ số giới hạn trên của các bậc (p, d, q). Mọi việc còn lại hãy để thuật toán tự giải quyết.

Cài đặt thư viện: `pip install pmdarima`

Gọi thư viện sử dụng: `from pmdarima import auto_arima`

### 2.6 Mô hình ARIMA mùa vụ (Seasonal ARIMA - SARIMA) <a class="anchor" id="2.6"></a>
Mô hình ARIMA rất tuyệt, nhưng để đưa tính thời vụ và các biến ngoại sinh vào mô hình có thể cực kỳ hiệu quả. Vì mô hình ARIMA giả định rằng chuỗi thời gian là cố định nên chúng ta cần sử dụng một mô hình khác.

**2.6.1 Kiểm tra yếu tố mùa vụ** <a class="anchor" id="2.6.1"></a>

Trong một số chuỗi thời gian thường xuất hiện yếu tố mùa vụ (seasonal). Việc tìm ra chu kì và quy luật mùa vụ sẽ giúp cho mô hình dự báo chuẩn xác hơn. Yếu tố mùa vụ cũng không phải là một trong những yếu tố quá khó để nhận biết. Chúng ta có thể dễ dàng phát hiện ra chúng thông qua đồ thị của chuỗi.

Có thể vẻ đồ thị của chuỗi dữ liệu ra để quan sát một cách tổng quát để kiểm tra rằng dữ liệu có tính chất mùa vụ hay không. Ngoài ra, chúng ta có thể sử dụng một phép phân rã mùa vụ (seasonal decompose) để trích lọc ra các thành phần cấu thành nên chuỗi bao gồm: xu hướng (trend), mùa vụ (seasonal), phần dư (residual).


**2.6.2 Công thức** <a class="anchor" id="2.6.2"></a>

Phương trình SARIMA (p, d, q)(P, D, Q) với các bậc $\text{(P, D, Q)}$ của yếu tố mùa vụ được trích xuất từ chuỗi ban đầu.

$$y_t = c + \sum_{n=1}^p{\alpha_n}{y_{t-n}} + \sum_{n=1}^q{\theta_n}{\epsilon_{t-n}} + \sum_{n=1}^P{\phi_n}{y_{t-sn}} + \sum_{n=1}^Q{\eta_n}{\epsilon_{t-sn}} + \epsilon_t $$

+ Nhập SARIMA (ARIMA theo mùa). Mô hình này rất giống với mô hình ARIMA, ngoại trừ việc có thêm một tập hợp các thành phần trung bình động và tự hồi quy.
+ Những độ trễ bổ sung này được bù đắp bởi tần suất theo mùa (ví dụ: 12 - hàng tháng, 24 - hàng giờ).
+ Các mô hình SARIMA cho phép phân biệt dữ liệu theo tần suất theo mùa, nhưng cũng theo sự khác biệt không theo mùa.
+ Việc biết tham số nào là tốt nhất có thể được thực hiện dễ dàng hơn thông qua các khung tìm kiếm tham số tự động,  như `pmdarima`.

**2.6.3 Xây dựng mô hình SARIMA** <a class="anchor" id="2.6.3"></a>

Để mô hình hiểu được chúng ta đang hồi quy trên mô hình SARIMA thì cần thiết lập tham số `seasonal=True` và chu kì của mùa vụ `m=12`.

Trong Python, có thể dùng hàm `seasonal_decompose`. Để dùng được hàm này, cần import vào: `from statsmodels.tsa.seasonal import seasonal_decompose`

### 2.7 Mô hình ARIMAX và SARIMAX <a class="anchor" id="2.7"></a>
**2.7.1 Giới thiệu** <a class="anchor" id="2.7.1"></a>

Mô hình SARIMA có thể được bổ sung thêm các biến giải thích. Điều này dẫn đến mô hình SARIMAX (Seasonal
Autoregressive Integrated Moving Average with Exogenous Variables).

**2.7.2 Công thức** <a class="anchor" id="2.7.2"></a>

$$d_t = c + \sum_{n=1}^p{\alpha_n}{d_{t-n}} + \sum_{n=1}^q{\theta_n}{\epsilon_{t-n}} + \sum_{n=1}^r{\beta_n}{\chi_{n_t}} + \sum_{n=1}^P{\phi_n}{d_{t-sn}} + \sum_{n=1}^Q{\eta_n}{\epsilon_{t-sn}} + \epsilon_t $$

+ Phiên bản cuối cùng của mô hình ARMA là mô hình ARIMAX và SARIMAX.
+ Các mô hình này tính đến các biến ngoại sinh, hay nói cách khác, sử dụng dữ liệu bên ngoài trong dự báo của chúng tôi.

**2.7.3 Xây dựng mô hình ARIMAX và SARIMAX** <a class="anchor" id="2.7.3"></a>

Mô hình SARIMAX được xây dựng bằng cách đưa một yếu tố dự đoán bên ngoài, còn được gọi là biến ngoại sinh (exogenous variable) vào mô hình `exogenous=df[['seasonal_index']]`

### 2.8 Xác định độ chính xác của mô hình <a class="anchor" id="2.8"></a>
Sau khi đã tìm ra được mô hình ARIMA tốt nhất. Chúng ta sẽ dự báo cho khoảng thời gian tiếp theo. Dự báo cho chuỗi thời gian khá đặc thù và khác biệt so với các lớp mô hình dự báo khác vì giá trị time step liền trước sẽ được sử dụng để dự báo cho time step liền sau. Do đó đòi hỏi phải có một vòng lặp liên tiếp dự báo qua các bước thời gian. Rất may mắn là hàm `predict()` đã tự động giúp ta thực hiện việc đó. Ta sẽ chỉ phải xác định số lượng phiên tiếp theo muốn dự báo là bao nhiêu.

Chúng ta biết rằng một mô hình có thể `fit` với tập huấn luyện (train set) nhưng chưa chắc đã tốt khi dự báo. Chính vì thế cần kiểm tra chất lượng của mô hình trên tập dự báo. Trong mô hình phân loại chúng ta thường quan tâm đến tỷ lệ chính xác `accuracy`, trong trường hợp mẫu mất cân bằng thì `precision`, `recall`, `f1` là những chỉ số đo lường độ chính xác khác được thay thế. Tuy nhiên với lớp mô hình dự báo thì sẽ sử dụng một tập hợp các tham số khác liên quan đến đo lường sai số giữa giá trị dự báo và giá trị thực tế. Đó là các chỉ số: **RMSE, MAE, MAPE**.

+ **RMSE – root mean square error:**

$$ \text{RMSE} = \sqrt{{\frac{1}{n} {\sum_{t=1}^n}{e_t^2}}}$$

+ **MAPE – mean absolute percentage error:**

$$ \text{MAPE} = \frac{1}{n}{\sum_{t=1}^n}\vert{\frac{e_t}{y_t}}\vert$$

+ **MAE – mean absolute error:**

$$ \text{MAE} = \frac{1}{n}{\sum_{t=1}^n}\vert{e_t}\vert$$

**Phân loại độ chính xác của mô hình dựa trên các giá trị MAPE dựa trên Lewis (1982).**
+ MAPE <= 10%: Highly accurate.
+ 10% < MAPE <= 20%: Quite Accurate.
+ 20% < MAPE <= 50%: Moderately Accurate
+ 50% <= MAPE: Not Accurate

**Tổng hợp các thước đo độ chính xác được sử dụng để đánh giá dự báo là:**
1. **MAPE** (Mean Absolute Percentage Error): Trung bình phần trăm trị tuyệt đối sai số. Chỉ số này cho biết giá trị dự báo sai lệch bao nhiêu phần trăm so với giá trị thực tế.
2. **ME** (Mean Error): Trung bình lỗi
3. **MAE** (Mean Absolute Error): Trung bình trị tuyệt đối sai số. Chính là khoảng cách theo norm chuẩn bậc 1 giữa giá trị dự báo và giá trị thực tế.
4. **MPE** (Mean Percentage Error): Trung bình phần trăm lỗi.
5. **RMSE** (Root Mean Squared Error): Phương sai hoặc độ lệch chuẩn của chuỗi dự báo so với thực tế.
6. **ACF1** (Lag 1 Autocorrelation of Error)
7. **corr** (Correlation between the Actual and the Forecast): Tương quan giữa thực tế và dự báo.
8. **minmax** (Min-Max Error)

Thông thường, nếu bạn đang so sánh các dự báo của hai chuỗi khác nhau, MAPE, Correlation và Min-Max Error có thể được sử dụng.

**Why not use the other metrics?**

Bởi vì chỉ có ba lỗi trên là lỗi tỷ lệ phần trăm (percentage errors) thay đổi trong khoảng từ 0 đến 1. Bằng cách đó, bạn có thể đánh giá mức độ tốt của dự báo bất kể quy mô của chuỗi.

Các số liệu lỗi khác là quantities. Điều đó có nghĩa là, RMSE là 100 cho một chuỗi có giá trị trung bình là 1000 thì tốt hơn RMSE là 5 cho chuỗi là 10. Vì vậy,chúng ta không thể sử dụng chúng để so sánh dự báo của hai chuỗi thời gian được chia tỷ lệ khác nhau.

## 3. Ưu và khuyết điểm của mô hình ARIMA <a class="anchor" id="3"></a>
[**Table of Contents**](#0.1)

### 3.1 Ưu điểm <a class="anchor" id="3.1"></a>
Trong đa số trường hợp, mô hình ARIMA cho kết quả dự báo ngắn hạn đáng tin cậy nhất trong các phương pháp dự báo. Hiện nay, mô hình dự báo ARIMA được sử dụng rộng rãi ở Việt Nam và trên thế giới cho các biến số kinh tế, tài chính, ...do tính dễ sử dụng, kết quả dự báo có độ chính xác tương đối cao (trừ trường hợp các biến có độ biến động quá lớn).

### 3.2 Khuyết điểm <a class="anchor" id="3.2"></a>
Hạn chế của mô hình ARIMA đó là số quan sát cần cho dự báo phải lớn (>= 50); đối với các biến số có biến động ngắn, ARIMA không hiệu quả vì không có tính chất phản ứng nhanh. Chỉ dùng để dự báo ngắn hạn và trong điều kiện tương đối ổn định; khả năng xây dựng kịch bản của mô hình ARIMA rất hạn chế.

## 4. Kết luận <a class="anchor" id="4"></a>
[**Table of Contents**](#0.1)

Trong phần này, tôi đã giới thiệu lí thuyết về Time Series, mô hình ARIMA và các biến thể của nó: ARIMA theo mùa (SARIMA) và ARIMAX sử dụng dữ liệu bên ngoài (đầu vào ngoại sinh) để cải thiện hiệu suất của mô hình ARIMA.

Bên cạnh đó, các khái niệm cần thiết và công thức toán cũng như cách mà mô hình đó hoạt động cũng được tôi đề cập đến. Qua đó, tôi đã cung cấp một cái nhìn tổng quan nhất về thuật toán ARIMA model.

## 5. Câu hỏi ôn tập <a class="anchor" id="5"></a>
[**Table of Contents**](#0.1)

**1. Phân tích chuỗi thời gian là gì?**
+ Chuỗi thời gian là một chuỗi các quan sát được thực hiện trong các khoảng thời gian xác định, thường là các khoảng thời gian bằng nhau. Phân tích chuỗi giúp chúng ta dự đoán các giá trị trong tương lai dựa trên các giá trị được quan sát trước đó. Trong Chuỗi thời gian, chúng tôi chỉ có 2 biến, thời gian và biến chúng tôi muốn dự báo.

**2. Tại sao và ở đâu thì Chuỗi thời gian được sử dụng?**
+ Dữ liệu chuỗi thời gian có thể được phân tích để trích xuất các số liệu thống kê có ý nghĩa và các đặc điểm khác. Nó được sử dụng trong ít nhất 4 kịch bản:
    + a) Dự báo kinh doanh.
    + b) Hiểu hành vi trong quá khứ.
    + c) Hoạch định tương lai.
    + d) Đánh giá thành tích hiện tại.
    
**3. Khi nào chúng ta không nên sử dụng Phân tích chuỗi thời gian?**
+ Chúng ta không cần áp dụng Chuỗi thời gian trong ít nhất 2 trường hợp sau:
    + a) Biến phụ thuộc (y) (được cho là thay đổi theo thời gian) là hằng số. Phương trình: y=f(x)=4, một đường thẳng song song với trục x(thời gian) sẽ luôn giữ nguyên.
    + b) Biến phụ thuộc (y) đại diện cho các giá trị có thể được biểu thị dưới dạng hàm toán học. Eq: sin(x), log(x), Đa thức, v.v. Do đó, chúng ta có thể trực tiếp nhận giá trị tại một số thời điểm bằng cách sử dụng chính hàm đó. Không cần dự báo.
    
**4. Các thành phần của Chuỗi thời gian là gì?**
+ Có 4 thành phần:
    + a) Xu hướng (Trend) - Chuyển động lên và xuống của dữ liệu theo thời gian trong một khoảng thời gian lớn. Eq: Đánh giá cao của Dollar so với rupee.
    + b) Tính thời vụ (Seasonality) - phương sai theo mùa. Eq: Doanh số bán kem chỉ tăng trong mùa hè.
    + c) Nhiễu hoặc bất thường (Noise or Irregularity) - Các đỉnh và đáy ở các khoảng thời gian ngẫu nhiên.
    + d) Chu kỳ (Cyclicity) - hành vi tự lặp lại sau một khoảng thời gian dài, như tháng, năm, v.v.
    
**5. Tính dừng (Stationarity) là gì?**
+ Trước khi áp dụng bất kỳ mô hình thống kê nào trên Chuỗi thời gian, chuỗi phải ổn định, có nghĩa là, trong các khoảng thời gian khác nhau,
    + a) Nó phải có giá trị trung bình không đổi.
    + b) Nó phải có phương sai hoặc độ lệch chuẩn không đổi.
    + c) Hiệp phương sai không nên phụ thuộc vào thời gian.

Xu hướng & Tính thời vụ là hai lý do khiến Chuỗi thời gian không ổn định & do đó cần phải được điều chỉnh.

**6. Tại sao Chuỗi thời gian (TS) cần phải dừng?**
+ Đó là vì những lý do sau:
    + a) Nếu một TS có một hành vi cụ thể trong một khoảng thời gian, thì có khả năng cao là trong một khoảng thời gian khác, nó sẽ có hành vi tương tự, với điều kiện TS là tĩnh. Điều này giúp dự báo chính xác.
    + b) b) Các lý thuyết & công thức toán học hoàn thiện hơn & dễ áp dụng hơn cho TS tĩnh.
    
**7. Các phép thử để kiểm tra xem một chuỗi có đứng yên hay không**
+ Có 2 cách để kiểm tra tính dừng của TS:
    + a) **Rolling Statistics** - Vẽ đồ thị trung bình động hoặc độ lệch chuẩn động (Plot the moving avg or moving standard deviation) để xem liệu nó có thay đổi theo thời gian hay không. Đó là một kỹ thuật hình ảnh.
    + b) **ADCF Test - Augmented Dickey–Fuller test** được sử dụng để cung cấp cho chúng ta các giá trị khác nhau có thể giúp xác định tính dừng. Giả thuyết Null nói rằng một TS là không cố định. Nó bao gồm Test Statistics & một số giá trị quan trọng đối với một số mức độ tin cậy. Nếu Test Statistics nhỏ hơn các giá trị tới hạn, chúng ta có thể bác bỏ giả thuyết không và nói rằng chuỗi này là dừng. Thử nghiệm ADCF cũng cho chúng ta giá trị p. Theo giả thuyết không, giá trị thấp hơn của p là tốt hơn.
    
**8. Mô hình ARIMA là gì?**
+ ARIMA(Auto Regressive Integrated Moving Average) là sự kết hợp của 2 mô hình AR(Auto Regressive) & MA(Moving Average). Nó có 3 hyperparameters - P(auto regressive lags),d(order of differentiation),Q(moving avg.) tương ứng xuất phát từ các thành phần AR, I & MA. Phần AR là mối tương quan giữa các khoảng thời gian trước và hiện tại. Để làm giảm noise, phần MA được sử dụng. Phần I liên kết các phần AR & MA với nhau.

**9. Làm thế nào để tìm giá trị P & Q cho ARIMA?**
+ Chúng tôi cần trợ giúp về các sơ đồ ACF(Auto Correlation Function) & PACF(Partial Auto Correlation Function). Đồ thị ACF & PACF được sử dụng để tìm giá trị P & Q cho ARIMA. Chúng ta cần kiểm tra xem giá trị nào trong trục x, đường biểu đồ giảm xuống 0 trong trục y lần đầu tiên.
    + Từ PACF(tại y=0), lấy P.
    + Từ ACF(tại y=0), lấy Q.
 
**10. Thử nghiệm ADCF là gì?**
+ Trong thống kê và kinh tế lượng, augmented Dickey–Fuller test (ADF) kiểm tra giả thuyết khống (null hypothesis) rằng một nghiệm đơn vị có mặt trong một mẫu chuỗi thời gian. Giả thuyết thay thế là khác nhau tùy thuộc vào phiên bản thử nghiệm nào được sử dụng, nhưng thường là trạng thái dừng hoặc trạng thái dừng theo xu hướng. Đây là một phiên bản mở rộng của phép thử Dickey–Fuller cho một tập hợp các mô hình chuỗi thời gian lớn hơn và phức tạp hơn.

Thống kê Augmented Dickey–Fuller (ADF) được sử dụng trong thử nghiệm, là một số âm. Nó càng phủ định, thì sự bác bỏ giả thuyết rằng có một nghiệm đơn vị ở một mức độ tin cậy nào đó càng mạnh mẽ.

p-value (0<=p<=1) phải càng thấp càng tốt. Các giá trị tới hạn ở các khoảng tin cậy khác nhau phải gần với giá trị Test statistics.
