import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import io

# Configure the page
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Backend API URL
API_URL = "http://localhost:5001/api"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .rating-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/status")
        return response.status_code == 200
    except:
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Check backend connection
    if not check_backend():
        st.error("❌ Backend server is not running. Please start the backend server first.")
        st.info("Run this command in your terminal: `python backend.py`")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", [
        "🏠 Home",
        "📊 Train Model", 
        "🔍 Single Review Analysis",
        "🎯 Movie Rating Analysis"
    ])
    
    # Get model status
    try:
        status_response = requests.get(f"{API_URL}/status")
        model_trained = status_response.json().get('is_trained', False)
    except:
        model_trained = False
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "📊 Train Model":
        show_train_model()
    elif app_mode == "🔍 Single Review Analysis":
        show_single_analysis(model_trained)
    elif app_mode == "🎯 Movie Rating Analysis":
        show_movie_analysis(model_trained)

def show_home():
    st.markdown("""
    ## Welcome to Movie Sentiment Analyzer!
    
    This AI-powered tool analyzes movie reviews and classifies them as **Positive** or **Negative** sentiment.
    You can also get overall movie ratings based on multiple reviews.
    
    ### Features:
    
    - **📊 Train Model**: Upload your dataset to train the sentiment analysis model
    - **🔍 Single Review Analysis**: Analyze individual movie reviews
    - **🎯 Movie Rating Analysis**: Get overall movie rating from multiple reviews
    
    ### How to use:
    
    1. Start by training the model with your dataset (CSV with 'review' and 'sentiment' columns)
    2. Once trained, you can analyze individual reviews or batches of reviews
    3. Get visualizations and detailed sentiment analysis
    
    ### Sample Dataset Format:
    
    Your CSV should have two columns:
    - `review`: The text of the movie review
    - `sentiment`: Either 'positive' or 'negative'
    """)
    
    # Show sample data
    sample_data = {
        'review': [
            "This movie was absolutely fantastic with great acting!",
            "Terrible film with poor acting and boring plot.",
            "Amazing cinematography and wonderful performances."
        ],
        'sentiment': ['positive', 'negative', 'positive']
    }
    
    st.subheader("Sample Data Format")
    st.dataframe(pd.DataFrame(sample_data))

def show_train_model():
    st.header("📊 Train Sentiment Analysis Model")
    
    st.info("""
    Upload a CSV file with movie reviews to train the model.
    Your CSV should have two columns: 'review' and 'sentiment'
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file to show preview
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} reviews.")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Show sentiment distribution
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                st.subheader("Sentiment Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Positive:** {sentiment_counts.get('positive', 0)}")
                    st.write(f"**Negative:** {sentiment_counts.get('negative', 0)}")
                with col2:
                    if len(df) > 0:
                        positive_pct = (sentiment_counts.get('positive', 0) / len(df)) * 100
                        negative_pct = (sentiment_counts.get('negative', 0) / len(df)) * 100
                        st.write(f"**Positive:** {positive_pct:.1f}%")
                        st.write(f"**Negative:** {negative_pct:.1f}%")
            
            # Train model button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model... This may take a few minutes..."):
                    try:
                        # Reset file pointer to beginning
                        uploaded_file.seek(0)
                        
                        # Send file to backend
                        files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
                        response = requests.post(f"{API_URL}/train", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Model trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{result['accuracy']:.2%}")
                            with col2:
                                st.metric("Training Samples", result['training_size'])
                            with col3:
                                st.metric("Test Samples", result['test_size'])
                            
                            # Show sentiment distribution from training
                            if 'sentiment_distribution' in result:
                                st.subheader("Training Dataset Distribution")
                                dist = result['sentiment_distribution']
                                st.write(f"**Positive:** {dist.get('positive', 0)}")
                                st.write(f"**Negative:** {dist.get('negative', 0)}")
                                
                        else:
                            error_msg = response.json().get('error', 'Unknown error')
                            st.error(f"❌ Error: {error_msg}")
                            st.info("Make sure your CSV has 'review' and 'sentiment' columns")
                            
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure you're uploading a valid CSV file")

def show_single_analysis(model_trained):
    st.header("🔍 Single Review Analysis")
    
    if not model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    review_text = st.text_area(
        "Enter movie review:",
        placeholder="Type your movie review here...",
        height=100
    )
    
    if st.button("Analyze Sentiment", type="primary") and review_text:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(f"{API_URL}/predict", 
                                       json={'review': review_text})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        if sentiment == 'positive':
                            st.markdown(f'<h2 class="sentiment-positive">👍 POSITIVE</h2>', 
                                      unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f'<h2 class="sentiment-negative">👎 NEGATIVE</h2>', 
                                      unsafe_allow_html=True)
                        
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    with col2:
                        st.subheader("Probabilities")
                        positive_prob = result['probabilities']['positive']
                        negative_prob = result['probabilities']['negative']
                        
                        col21, col22 = st.columns(2)
                        with col21:
                            st.metric("Positive", f"{positive_prob:.2%}")
                        with col22:
                            st.metric("Negative", f"{negative_prob:.2%}")
                    
                    st.subheader("Review Analysis")
                    st.info(f"**Review:** {review_text}")
                    
                    if 'emotions' in result:
                        st.subheader("Emotional Fingerprint")

                        emotions = result['emotions']

                        emotion_df = pd.DataFrame({
                            'Emotion': list(emotions.keys()),
                            'Score': list(emotions.values())
                        })

                        fig = px.pie(
                            emotion_df,
                            names='Emotion',
                            values='Score',
                            color='Emotion',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textinfo='label+percent')
                        st.plotly_chart(fig, use_container_width=True)

                        emotion_percent_df = emotion_df.copy()
                        emotion_percent_df['Percentage'] = (emotion_percent_df['Score'] * 100).round(2).astype(str) + '%'
                        st.dataframe(emotion_percent_df[['Emotion', 'Percentage']], use_container_width=True, hide_index=True)

                    if 'reviewer_style' in result:
                        st.subheader("Reviewer Style Detector")
                        style = result['reviewer_style']

                        col31, col32, col33 = st.columns(3)
                        with col31:
                            st.metric("Style", style['style'].title())
                        with col32:
                            st.metric("Tone", style['tone'].title())
                        with col33:
                            st.metric("Focus", style['focus'].title())

                        st.info(f"**Why this style?** {style['explanation']}")

                    if 'mismatch_insight' in result:
                        st.subheader("Sentiment vs Emotion Insight")
                        mismatch = result['mismatch_insight']

                        if mismatch['mismatch']:
                            st.warning(f"**{mismatch['type'].title()}**: {mismatch['summary']}")
                        else:
                            st.success(mismatch['summary'])
                        st.write(mismatch['detail'])

                    if 'recommendations' in result:
                        st.subheader("What To Watch Next")
                        st.caption("Generated from the emotional and tonal pattern of this review only.")

                        for recommendation in result['recommendations']:
                            with st.container():
                                rec_col1, rec_col2 = st.columns([1, 3])
                                with rec_col1:
                                    if recommendation.get('poster_url'):
                                        st.image(recommendation['poster_url'], use_container_width=True)
                                    else:
                                        st.caption("Poster unavailable")
                                with rec_col2:
                                    st.markdown(f"### {recommendation['title']} ({recommendation['year']})")
                                    st.write(f"**Match Score:** {recommendation['match_score']:.1%}")
                                    st.write(f"**Why it fits:** {recommendation['reason']}")
                                    st.write(f"**Overview:** {recommendation['overview']}")
                    
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    st.error(f"❌ Error: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

def show_movie_analysis(model_trained):
    st.header("🎯 Movie Rating Analysis")
    
    if not model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    st.info("Enter multiple reviews for the same movie to get an overall rating")
    
    # Input for multiple reviews
    reviews = st.text_area(
        "Enter reviews (one per line):",
        placeholder="Enter each review on a new line...\nExample:\nThis movie was great!\nAmazing acting!\nPoor storyline...",
        height=200
    )
    
    if st.button("Calculate Movie Rating", type="primary") and reviews:
        review_list = [review.strip() for review in reviews.split('\n') if review.strip()]
        
        if len(review_list) < 1:
            st.warning("Please enter at least 1 review")
            return
            
        with st.spinner("Analyzing movie reviews..."):
            try:
                response = requests.post(f"{API_URL}/analyze-movie", 
                                       json={'reviews': review_list})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display overall rating
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="rating-box">', unsafe_allow_html=True)
                        st.metric("Overall Rating", f"{result['movie_rating']}/10")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Positive Reviews", 
                                 f"{result['positive_reviews']}/{result['total_reviews']}",
                                 f"{result['positive_percentage']:.1f}%")
                    
                    with col3:
                        st.metric("Negative Reviews", 
                                 f"{result['negative_reviews']}/{result['total_reviews']}",
                                 f"{result['negative_percentage']:.1f}%")
                    
                    # Sentiment distribution
                    st.subheader("Sentiment Distribution")
                    col4, col5 = st.columns(2)
                    
                    with col4:
                        if result['positive_reviews'] > 0:
                            st.progress(result['positive_percentage']/100, text=f"Positive: {result['positive_percentage']:.1f}%")
                        if result['negative_reviews'] > 0:
                            st.progress(result['negative_percentage']/100, text=f"Negative: {result['negative_percentage']:.1f}%")
                    
                    with col5:
                        st.metric("Average Confidence", f"{result['average_confidence']:.2%}")
                    
                    # Individual predictions
                    st.subheader("Individual Review Analysis")
                    for i, (review, prediction) in enumerate(zip(review_list, result['individual_predictions'])):
                        with st.expander(f"Review {i+1}: {review[:50]}...", expanded=False):
                            sentiment_icon = "👍" if prediction['sentiment'] == 'positive' else "👎"
                            st.write(f"**Sentiment:** {sentiment_icon} {prediction['sentiment'].upper()}")
                            st.write(f"**Confidence:** {prediction['confidence']:.2%}")
                            st.write(f"**Review:** {review}")
                
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    st.error(f"❌ Error: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
