NQ Futures Historical Dataset
=================================== 

NOTE: Only intraday-bars with trading volume are included. Bars with zero volume are excluded.

Timeframes : 1-minute, 5-minutes, 30-minutes, 1-hour, 1-day
(individual contract data is only available in 1-min and 1-day timeframes).


Dates : 
Continuous Date - First Date - 2008-01-02   |   Last Date 2025-05-09
Individual Contracts : First Contract - NQZ08   |    Last Contract NQZ24

Adjustments
------------

We provide both continuous data (ie a single series of chained contracts) as well as individual contract data (ie data for each futures contract delivered in a single file per contract).

We provide three types of continuous data :
Unadjusted - actual historic traded prices with no adjustments 
Absolute-Adjusted - prices adjusted for price jumps on contract roll date by using the absolute value of the price movement
Ratio-Adjusted  - prices adjusted for price jumps on contract roll date by using the percentage movement.

More details on the adjustments are available at  https://firstratedata.com/about/price_adjustment#futures

Format
-------
Intraday Data is in the format : {DateTime (yyyy-MM-dd HH:mm:ss), Open, High, Low, Close, Volume}   
Daily (1day) data is in the format : {DateTime (yyyy-MM-dd HH:mm:ss), Open, High, Low, Close, Volume, Open Interest}   

- Volume Numbers are in units of futures
- Timestamps run from the start of the period (eg 1min bars stamped 09:30 run from 09:30.00 to 09:30.59)
- Times with zero volume are omitted (thus gaps in the data sequence are when there have been no trades)


Updates
-------
Data is updated - (available by 11pm of the current trading day).   

Notes
-----
 
- Timezone is US Eastern Time    
- Excel will usually fail to open large files directly. 
  To use the data in Excel, open the files using notepad and then break into smaller files or copy/paste into Excel 
- Data license is available at https://firstratedata.com/about/license
 
  
* update times are approximate and may experience small delays in the event of a server or network outage.
 
 
___________________________
copyright FirstRateData.com