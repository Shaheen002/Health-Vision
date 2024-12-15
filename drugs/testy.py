from cryptography.fernet import Fernet 
stri = "pqr@xyz.com"
passy = "pancere"
key = Fernet.generate_key()
cipher_text = Fernet(key)
key_str = key.decode('utf-8')
encrypted_string = cipher_text.encrypt(stri.encode())
fv = encrypted_string.decode('utf-8')
encrypted_string2 = cipher_text.encrypt(passy.encode())
fv2 = encrypted_string2.decode('utf-8')
import dbconn  
connection = dbconn.DBConnection()
conn = connection.createConnection(u_name='root',passwd = '1234',db = 'healthvis')
cursor = conn.cursor()
cursor.execute("INSERT INTO LOGIN_DETAILS VALUES('{}','{}','{}')".format(fv,fv2,key_str))
conn.commit()
#bytes_object = A.encode('utf-32')
key = key_str.encode('utf-8')
bytes_object = fv.encode('utf-8')
cipher_suite = Fernet(key)
decrypted_string = cipher_suite.decrypt(bytes_object).decode()
print(decrypted_string)


