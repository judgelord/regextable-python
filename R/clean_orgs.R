# a function to clean org names
clean_orgs <- . %>%
  str_rm_all("\\(.*| Action Fund|^The |of the United States| - .*|:.*|Mass Comment Campaign of the |/.*| USA$|\\.com") %>%
  str_squish() %>%
  str_to_title() %>%
  str_replace(".*Sierra.*", "Sierra Club")%>% 
  str_replace(".*Nrdc.*|Natural Resources Defense Council.*", "NRDC")%>% 
  str_replace(".*Center For Biological Diversity.*|Cneter For Biological Diversity", "Center For Biological Diversity")%>% 
  str_replace(".*World Wildlife Fund.*|.WWF", "World Wildlife Fund")%>%
  str_replace(".*Greenpeace.*", "Greenpeace")%>%
  str_replace(".*Pew .*","Pew") %>% 
  str_replace(".*Audubon.*", "Audubon") %>% 
  str_replace(".*Credo.*", "Credo")%>%
  str_replace(".*Defenders Of Wildlife.*", "Defenders Of Wildlife")%>%
  str_replace(".*Friends Of The Earth.*", "Friends Of The Earth")%>%
  str_replace(".*Earthjustice.*|.*Earth Justice.*", "Earthjustice")%>%
  str_replace(".*Defenders Of Wildlife.*", "Defenders Of Wildlife")%>%
  str_replace(".*American Heart Association.*", "American Heart Association")%>%
  str_replace(".*Audubon.*", "Audubon")%>%
  str_replace(".*Wildlife Conservation Society.*", "Wildlife Conservation Society")%>%
  str_replace(".*Oceana.*", "OCEANA")%>%
  str_replace(".*Planned Parenthood.*", "Planned Parenthood")%>%
  
  str_replace(".*Moveon.*|.*Move On.*", "credo")%>%
  str_replace("Nat'l", "National")%>% 
  str_replace(".*Ifaw.*", "INTERNATIONAL FUND FOR ANIMAL WELFARE")%>% 
  str_replace(".*Humane Society.*|HSUS.*|.*\\bHSI\\b.*", "Humane Society")%>%
  str_replace("Moms Rising|Momsrising|Mom's Rising|Momsrisiing", "Moms Rising")%>%
  str_replace("Care 2|Care2", "Care2")%>%
  str_replace("Preventobesity.*", "American Heart Association")%>%
  str_replace("Axess.*|.*Axcess\\b.*", "AXCESS FINANCIAL")%>%
  str_replace(".*American Petroleum Institute.*|.*\\bApi\\b.*|Joe Jansen", "American Petroleum Institute") %>%
  str_replace(".*Consumer Energy Alliance.*", "Consumer Energy Alliance") %>%
  str_replace("Saveourenvironment.org|Save Our Environment$", "Partnership Project") %>% 
  str_replace("^None$|^Na$|^Self$|^Private Citizen$|^Citizen$|^Individual$|^Private Individual$|^Personal$|^Retired$|^Myself$|^US Citizen|^U.S. Citizen$|^Not Applicable|^Me$|^Student$|^Mr.$|^Ms.$|^Mrs.$|^No Organization", "Individual") %>% 
  str_squish() %>%
  str_to_title()

