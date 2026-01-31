import yagmail
from loguru import logger
import smtplib
import ssl
from email.mime.text import MIMEText
from email.header import Header
import os


def send_yagmail(exp_name):
	smtp_servers = ['smtp.qq.com']
	emails = []
	auth_codes = []
	sender_id = 0
	receiver_id = 0
	smtp_server = smtp_servers[sender_id]
	sender = emails[sender_id]
	receiver = emails[receiver_id]
	auth_code = auth_codes[sender_id]
	yagmail_server = yagmail.SMTP(user=sender, password=auth_code, host=smtp_server)
	email_title = ["Training Completed"]
	email_content = [f"The training process of experiment {exp_name} on the remote server is completed"]
	file = None
	if os.path.exists('nohup.out'):
		file = 'nohup.out'
	elif os.path.exists(f'{exp_name}.log'):
		file = f'{exp_name}.log'
	yagmail_server.send(to=receiver, subject=email_title, contents=email_content, attachments=file)
	yagmail_server.close()
	logger.info("Email Sent!")