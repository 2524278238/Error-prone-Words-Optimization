from src.utils.common import *

a1=readline('/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.en')
a2=readline('/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.en2')[:700000]
a3=readline('/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.en3')


with open('/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.de','w') as f:
    for i in a1+a2+a3:
        f.write(i+'\n')





