#!/bin/bash

get_action_name()
{
  MYFILENAME=`basename "$0"`
  ACTIONHARDWARE=`echo $MYFILENAME | sed 's/sendmail_//' | sed 's/.sh//'`
  echo Selected model based on folder name: $MODELNAME
}

send_mail()
{
  subject="Inference $ACTIONHARDWARE"
  body="Inference $ACTIONHARDWARE"
  from="intelnuccdl@gmail.com"
  echo -e "Subject:${subject}\n${body}" | sendmail -f "${from}" -t "${USEREMAIL}"
}

echo "#==============================================#"
echo "# CDLEML Process Sendmail"
echo "#==============================================#"

# Constant Definition
USEREMAIL=alexander.wendt@tuwien.ac.at
#HARDWARENAME=IntelNUC

#ACTION=TF2_start

#Extract model name from this filename
get_action_name

#Send mail
send_mail
