import streamlit as st
import preprocessor,helper,emoji_sentiment,text_sentiment1
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()

    # df['emoji'] = df['user'].apply(emoji_sentiment.get_emojis)
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            emoji_font_path = "C:\\Users\\Siddhartha\\Downloads\\Noto_Emoji\\NotoEmoji-VariableFont_wght.ttf"
            prop = fm.FontProperties(fname=emoji_font_path, size=14)
            fig,ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(emoji_df['count'].head(),labels=emoji_df['emoji'].head(),autopct="%0.2f",textprops={'fontproperties': prop})
            
            for autotext in autotexts:
                autotext.set_fontproperties(fm.FontProperties(size=12))
            st.pyplot(fig)
            

        # emoji sentiment score
        st.title("Emoji Sentiment Score")
        pos,neg,neu,score = emoji_sentiment.get_score(emoji_df)
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots()
            ax.pie([pos,neu,neg],labels=['Positive','Neutral','Negative'], autopct = "%0.2f")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Emoji Score")
            st.write(score)

        st.title("Overall Sentiment Score")
        sentiments = text_sentiment1.get_score(selected_user, df)
        
        def label_sentiment(sentiment):
            if sentiment in [3, 4]:
                return 'Positive'
            elif sentiment in [0, 1]:
                return 'Negative'
            else:
                return 'Neutral'

        # Add the sentiment_label column
        df['sentiment_label'] = df['sentiment'].apply(label_sentiment)
        # sentiment_df['sentiment'] = np.where(sentiment_df['sentiment'].str.contains('1 star|2 stars'), 'Negative',np.where(sentiment_df['label'].str.contains('3 stars'), 'Neutral','Positive'))
        sentiment_counts = df['sentiment_label'].value_counts()

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color='skyblue', ax=ax)
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        plt.title('Sentiment Analysis of WhatsApp Messages')
        st.pyplot(fig)
