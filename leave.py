jan=31
feb=28
mar=31
apr=30
may=31
jun=30
jul=31
aug=31
sep=30
oct=31
nov=30
dec=31

#leave year ends 30th Sept
days_in_year=360.0
annual_leave=27.5

leave_to_apr23=(annual_leave/days_in_year)*(oct+nov+dec+jan+feb+mar+23)
days_carried_over=10.0

leave_to_take=leave_to_apr23+days_carried_over

##################

days_taken=0.5+3.0+4.0+0.5+1.0+16.0+0.5

leave_to_take-days_taken

#gives 0.16 days remaining

holiday_allowance_to_30th_sept=27.5-leave_to_apr23
#gives 11.84 days 

youself_remaining_leave_this_year=14.0 #days

youself_remaining_leave_this_year-holiday_allowance_to_30th_sept
#gives 2.16 days remaining

#I believe this reflects the fact that I carried over 12 days into this year

