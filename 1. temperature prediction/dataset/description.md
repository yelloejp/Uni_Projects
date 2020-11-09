It collects temperature data every 10 mins from sensors which are inside or outside. 

1. Train dataset 
X00 ~ X39(from National Weather Service), Y00 ~ Y18(from sensors)
    - X00, X07, X28, X31, X32 : temperature
    - X01, X06, X22, X27, X29 : atmospheric pressure
    - X02, X03, X18, X24, X26 : wind speed
    - X04, X10, X21, X36, X39 : daily cumulative precipitation
    - X05, X08, X09, X23, X33 : sea level pressure
    - X11, X14, X16, X19, X34 : daily cumulative insolation
    - X12, X20, X30, X37, X38 : humidity
    - X13, X15, X17, X25, X35 : wind direction
    - Y00 ~ Y17 : observed temperature for each sensor
    - Y18 : target variable
    
2. Test dataset 
X00 ~ X39(from National Weather Service)
    - X00, X07, X28, X31, X32 : temperature
    - X01, X06, X22, X27, X29 : atmospheric pressure
    - X02, X03, X18, X24, X26 : wind speed
    - X04, X10, X21, X36, X39 : daily cumulative precipitation
    - X05, X08, X09, X23, X33 : sea level pressure
    - X11, X14, X16, X19, X34 : daily cumulative insolation
    - X12, X20, X30, X37, X38 : humidity
    - X13, X15, X17, X25, X35 : wind direction
