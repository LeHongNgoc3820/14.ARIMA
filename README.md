AN INTRODUCTION TO ARIMA MODEL THEORY

### Nội dung
1. Giới thiệu
2. Dự đoán - thuật toán
3. Áp dụng auto arima xây dựng mô hình ARIMA

## 1. Giới thiệu
+ ARIMA (Auto Regressive Integrated Moving Average) là một lớp mô hình dự đoán phổ biến và linh hoạt sử dụng thông tin lịch sử để đưa ra dự đoán. Loại mô hình này là một kỹ thuật dự đoán cơ bản có thể được sử dụng làm nền tảng cho các mô hình phức tạp hơn.
+ Có các mô hình ARIMA theo mùa (Seasonal) và không theo mùa (Non-seasonal) có thể được sử dụng để dự đoán.

### Ứng dụng
+ Phân tích chuỗi thời gian (Time series analysis) có thể được sử dụng trong vô số ứng dụng kinh doanh để dự báo số lượng trong tương lai và giải thích các mô hình lịch sử của nó. Ví dụ các trường hợp ứng dụng:
    + Giải thích các mô hình theo mùa trong bán hàng.
    + Dự đoán số lượng khách hàng đến hoặc đi dự kiến.
    + Ước tính ảnh hưởng của một sản phẩm mới ra mắt về số lượng bán.
    + Phát hiện các sự kiện bất thường và ước tính mức độ ảnh hưởng của chúng.
    
## 2. Thuật toán
+ Dữ liệu chuỗi thời gian (Time Series Data) là dữ liệu thử nghiệm đã được quan sát tại các thời điểm khác nhau theo thời gian (thường là khoảng cách đều nhau, như một lần mỗi ngày/mỗi tháng/mỗi quý, ...)
> **Ví dụ:** Dữ liệu bán vé máy bay mỗi ngày là một chuỗi thời gian hay tốc độ tăng trưởng kinh tế hàng năm, ...

+ Time series có một số tính năng chính như xu hướng (trend), tính thời vụ (seasonal) và nhiễu (noise).
+ Công việc của chúng ta là phân tích các tính năng này của tập dữ liệu time series và sau đó áp dụng mô hình để dự đoán trong tương lai.
+ Tuy nhiên, một loạt các sự kiện có yếu tố thời gian không tự động biến nó thành chuỗi thời gian, chẳng hạn như các ngày xảy ra thảm hoạ hàng không, với khoảng cách thời gian ngẫu nhiên thì không phải là chuỗi thời gian. Loại quy trình ngẫu nhiên này được gọi là quá trình điểm (point process).
+ Dự đoán:
    + Dự đoán rất khó, đặc biệt là dự đoán về tương lai
    + Dự báo là quá trình đưa ra dự đoán về tương lai, dựa trên dữ liệu trong quá khứ và hiện tại
    
+ Trong mô hình ARIMA có 3 tham số được sử dụng để giúp mô hình hoá các khía cạnh chính của một chuỗi thời gian: seasonality, trend và noise. Các tham số này được gắn nhãn lần lượt là p, d và q.
+ Một mô hình ARIMA thường được ghi là ARIMA(p,d,q).
+ **Trong đó:**
    + `p`: là tham số kết hợp với khía cạnh tự động hồi quy của mô hình (auto-regressive aspect - AR), kết hợp các giá trị trong quá khứ mang tính chất lâu dài (giá trị quan sát hiện tại phụ thuộc vào các giá trị trước đó). (Tổng trọng số của các giá trị độ trễ của series)
    > **Ví dụ**: Dự báo rằng nếu trời mưa nhiều trong vài ngày qua, có thể cho biết ngày mai trời sẽ mưa.
    + `d` (difference): là tham số kết hợp với phần tích hợp của mô hình (integrated part - I), nó ảnh hưởng đến lượng chênh lệch áp dụng cho một chuỗi thời gian (Sự khác biệt của time series)
    > **Ví dụ**: Dự báo rằng lượng mưa ngày mai sẽ tương tự như lượng mưa ngày hôm nay, nếu lượng mưa hàng ngày tương tự trong vài ngày qua.
    + `q`: là tham số liên quan đến phần trung bình động của mô hình (moving average part - MA, các số liệu phụ thuộc nhau trong một khoảng thời gian ngắn). (Tổng số các lỗi dự báo bị trễ của series)
    
### Ghi chú:
+ ARIMA được ứng dụng thường xuyên cho các dãy dữ liệu theo thời gian ổn định (Stationary time series).
+ Trong thống kê, dữ liệu thời gian ổn định là dữ liệu mà các chỉ số thống kế không đổi (trung bình, phương sai, hệ số tương quan, ...) theo thời gian.
+ Khi trung bình và phương sai có xu hướng biến chuyển theo thời gian thì sẽ có dữ liệu bất ổn định (non-stationary time series). Lúc này chúng ta phải tính bậc (order) của `d` để có được dữ liệu ổn định.
+ Nếu mô hình có thành phần theo mùa, chúng ta sử dụng mô hình ARIMA theo mùa (SARIMA). Trong trường hợp đó, sẽ có một bộ tham số khác: P, D và Q mô tả các liên kết tương tự như p, d và q nhưng tương ứng với các thành phần theo mùa của mô hình (Seasonal Model).

### Thuộc tính và loại của Series
+ **Trend**: Tăng hoặc giảm dài hạn trong dữ liệu. Có thể được xem như là một độ dốc - slope (không phải là tuyến tính) gần như đi xuyên qua dữ liệu.
+ **Seasonality**: Một chuỗi thời gian được cho là thời vụ khi nó bị ảnh hưởng bởi các yếu tố theo mùa (giờ trong ngày, tuần, tháng, năm, ...). Tính thời vụ có thể được quan sát với các mẫu chu kỳ (cyclical patterns) có tần số cố định (fixed frequency).
+ **Cyclicity**: Một chu kỳ xảy ra khi dữ liệu biểu hiện tăng và giảm không có tần số cố định. Những biến động này thường là do điều kiện kinh tế và thường liên quan đến "Business cycle". Thời gian của những biến động này thường ít nhất là 2 năm.
+ **Residuals**: Mỗi chuỗi thời gian có thể được phân tách thành hai phần:
    + **Forecast**: bao gồm một hoặc một số giá trị dự báo (forecasted values)
    + **Residuals**: sự khác biệt giữa một quan sát (observation) và giá trị được dự đoán của nó ở mỗi time step.
    
    $$ \text{Value of series at time t = (Predicted value at time t) + (Residual at time t)} $$

### Công thức ARIMA
Mô hình ARIMA là viết tắt của "Auto-Regressive Integrated Moving Average" và có thể được chia thành **AR, I, MA**.

### a. Autoregressive Component - AR(p)
Thành phần tự hồi quy (autoregressive component) của mô hình ARIMA được đại diện bởi AR(p), với tham số `p` xác định số chuỗi bị trễ mà chúng tôi sử dụng.

$$y_t = c + \sum_{n=1}^p{\alpha_i}{y_{t-n}} + \epsilon_t $$

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

### Moving Average - MA(q)
"This component is not a rolling average, but rather the lags in the white noise."

**MA(q)**
+ MA(q) là mô hình trung bình động và q là số thuật ngữ lỗi dự báo bị trễ trong dự đoán.
+ Trong mô hình MA(1), dự báo của chúng tôi là một số hạng không đổi cộng với số hạng nhiễu trắng trước đó nhân với một số nhân, được cộng với thuật ngữ nhiễu trắng hiện tại.
+ Đây chỉ là xác suất + thống kê đơn giản vì chúng tôi đang điều chỉnh dự báo của mình dựa trên các thuật ngữ white noise trước đó.

### ARMA and ARIMA Models
ARMA and ARIMA architectures are just the AR (Autoregressive) and MA (Moving Average) components put together.

**ARMA**
+ Mô hình ARMA là một hằng số cộng với tổng độ trễ AR và hệ số nhân của chúng, cộng với tổng độ trễ MA và hệ số nhân của chúng cộng với nhiễu trắng. Phương trình này là cơ sở của tất cả các mô hình tiếp theo và là khuôn khổ cho nhiều mô hình dự báo trên các lĩnh vực khác nhau.

**ARIMA**

$$Y_t = {\beta_2} + {\omega_1}{\epsilon_{t-1}} + {\omega_2}{\epsilon_{t-2}} + ... + {\omega_q}{\epsilon_{t-q}} + {\epsilon_t}$$

+ Mô hình ARIMA là một mô hình ARMA với bước tiền xử lý được bao gồm trong mô hình mà chúng tôi đại diện bằng cách sử dụng I(d).
+ I(d) là thứ tự khác biệt, là số phép biến đổi cần thiết để làm cho dữ liệu đứng yên (data stationary).
+ Vì vậy, một mô hình ARIMA chỉ đơn giản là một mô hình ARMA trên chuỗi thời gian khác nhau.

### SARIMA, ARIMAX, SARIMAX Models
The ARIMA model is great, but to include seasonality and exogenous variables in the model can be extremely powerful. Since the ARIMA model assumes that the time series is stationary, we need to use a different model.

**SARIMA**

$$y_t = c + \sum_{n=1}^p{\alpha_n}{y_{t-n}} + \sum_{n=1}^q{\theta_n}{\epsilon_{t-n}} + \sum_{n=1}^P{\phi_n}{y_{t-sn}} + \sum_{n=1}^Q{\eta_n}{\epsilon_{t-sn}} + \epsilon_t $$

+ Nhập SARIMA (ARIMA theo mùa). Mô hình này rất giống với mô hình ARIMA, ngoại trừ việc có thêm một tập hợp các thành phần trung bình động và tự hồi quy.
+ Những độ trễ bổ sung này được bù đắp bởi tần suất theo mùa (ví dụ: 12 - hàng tháng, 24 - hàng giờ).
+ Các mô hình SARIMA cho phép phân biệt dữ liệu theo tần suất theo mùa, nhưng cũng theo sự khác biệt không theo mùa.
+ Việc biết tham số nào là tốt nhất có thể được thực hiện dễ dàng hơn thông qua các khung tìm kiếm tham số tự động,  như `pmdarima`.

**ARIMAX and SARIMAX**

$$d_t = c + \sum_{n=1}^p{\alpha_n}{d_{t-n}} + \sum_{n=1}^q{\theta_n}{\epsilon_{t-n}} + \sum_{n=1}^r{\beta_n}{\chi_{n_t}} + \sum_{n=1}^P{\phi_n}{d_{t-sn}} + \sum_{n=1}^Q{\eta_n}{\epsilon_{t-sn}} + \epsilon_t $$

+ Phiên bản cuối cùng của mô hình ARMA là mô hình ARIMAX và SARIMAX.
+ Các mô hình này tính đến các biến ngoại sinh, hay nói cách khác, sử dụng dữ liệu bên ngoài trong dự báo của chúng tôi.


### Variation
+ Một trong những tính năng quan trọng nhất của một time series là variation (biến thể). Biến thể là các mẫu trong time series. Time series có các mẫu lặp lại trong khoảng thời gian đã biết và cố định được cho là có tính thời vụ (seasonality). Tính thời vụ là thuật ngữ chung cho các biến thể lặp lại định kỳ trong dữ liệu. Các biến thể có 4 loại: theo mùa, theo chù kỳ, xu hướng và biến động không đều (Seasonal, Cyclic, Trend và Irregular fluctuations).
+ Biến động theo mùa (Seasonal Variation) thường được định nghĩa là biến thể hàng năm trong kỳ, chẳng hạn như doanh số bán áo tắm thấp hơn vào mùa động và cao hơn vào mùa hè.
+ Biến thiên tuần hoàn (Cyclic Variation) là một biến thể xảy ra tại các khoảng thời gian cố định khác, trảng hạn như biến đổi nhiệt độ hàng ngày.
+ Cả hai biến thể theo Seasonal và Cyclic là ví dụ về tính thời vụ trong tập dữ liệu chuỗi thời gian.
+ Xu hướng (Trend) là những thay đổi dài hạn ở mức trung bình, liên quan đến số lượng mẫu quan sát.

### Decomposition (phân tích)
+ Một chuỗi thời gian là sự kết hợp của các thành phần sau:
    + Trend: chuyển động lên hoặc xuống của đường cong dài hạn (long term)
    + Seasonal component: Thành phần theo mùa
    + Residuals

## 3. Áp dụng auto arima xây dựng mô hình ARIMA

Cài đặt thư viện: `pip install pmdarima`

Gọi thư viện sử dụng: `from pmdarima import auto_arima`

### AIC (The Akaike information criterion)
+ AIC là một bộ ước lượng chất lượng tương đối của các mô hình thống kê cho một tập dữ liệu nhất định. Cung cấp một tập hợp các mô hình cho dữ liệu, AIC sẽ ước tính chất lượng của từng mô hình, liên quan đến từng mô hình khác.
+ Giá trị AIC cho phép so sánh mô hình phù hợp với dữ liệu và tính đến độ phức tạp của mô hình, vì vậy các mô hình phù hợp hơn trong khi sử dụng ít tính năng hơn sẽ nhận được điểm AIC tốt hơn (thấp hơn) các mô hình tương tự sử dụng nhiều tính năng hơn.
+ Thư viện ARIMA cho phép ta nhanh chóng thực hiện Grid Search và tạo ra một model object mà ta có thể fit training data.
+ Thư này này chứa auto arima function cho phép chúng ta thiết lập một loạt các giá trị p, d, q, P, D< Q và m; sau đó phù hợp với các mô hình cho tất cả các kết hợp có thể. Sau đó, mô hình sẽ giữ kết hợp có trị AIC tốt nhất.
