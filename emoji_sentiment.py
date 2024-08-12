import advertools as adv
import pandas as pd

lexicon = pd.read_csv('C:\\Users\\Siddhartha\\OneDrive\\Desktop\\python project\\mini\\emoji_lexicon.csv')


lexicon_dict = {row['Char']:[row['Pos'],row['Neg'],row['Neut'],row['Sentiment_score']] for _,row in lexicon.iterrows()}
    
# print(lexicon_dict)
def get_score(df):
    
    pos,neu,neg,score,count=0,0,0,0,0
    
    for _,row in df.iterrows():
        emoji = row['emoji']
        count+=row['count']
        pos_score,neg_score,neu_score = lexicon_dict[emoji][0],lexicon_dict[emoji][1],lexicon_dict[emoji][2]
        score += lexicon_dict[emoji][3]*row['count']
        if pos_score>neg_score:
            if pos_score>neu_score:
                pos+=1
            else:
                neu+=1
        else:
            if neg_score>neu_score:
                neg+=1
            else:
                neu+=1
    
    score /= count

    return pos,neu,neg,score