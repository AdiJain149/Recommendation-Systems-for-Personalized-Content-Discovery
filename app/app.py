import os
import streamlit as st
import requests

# Configure high-level professional viewport layouts
st.set_page_config(
    page_title="Netflix Movie Recommendation System",
    page_icon="🍿",
    layout="centered"
)

# Hardcode the target microservice network URL pointing to our active FastAPI instance
API_URL = "http://127.0.0.1:8000/api/v1/recommend"

st.title("🍿 Netflix Recommendation Dashboard")
st.markdown(
    """
    This real-time dashboard communicates with a high-performance **FastAPI microservice** serving recommendations from a matrix factorization model trained on the **Netflix Prize Dataset**.
    """
)
st.markdown("---")

# Section 1: User Profile Configuration Input
st.subheader("👤 Target Profile Analysis")
user_id_input = st.number_input(
    "Enter a valid Netflix User ID:",
    min_value=1,
    value=305344,  # Default out-of-the-box target from Phase 3 EDA
    step=1,
    help="Type in a user ID to analyze historical preferences and generate predictions."
)

recommendation_count = st.slider(
    "Number of recommendations to retrieve (K):",
    min_value=1,
    max_value=20,
    value=10,
    step=1
)

# Section 2: Inference Triggers and State Tracking
if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Executing inference matrix lookup on backend service..."):
        try:
            # Dispatch network request to the live FastAPI gateway
            response = requests.get(
                f"{API_URL}/{user_id_input}",
                params={"k": recommendation_count},
                timeout=5.0  # Safe network timeout limit
            )
            
            # Check for standard server side execution problems
            if response.status_code == 200:
                payload = response.json()
                recommendations = payload.get("recommendations", [])
                
                if not recommendations:
                    st.warning("⚠️ Inference completed, but no recommendation candidates returned for this user configuration.")
                else:
                    st.success(f"🎉 Successfully parsed {len(recommendations)} tailored recommendations!")
                    
                    # Convert raw JSON response into a clean, presentation-ready table
                    formatted_data = []
                    for idx, rec in enumerate(recommendations):
                        formatted_data.append({
                            "Rank": idx + 1,
                            "Movie ID Code": f"🎬 Movie #{rec['movie_id']}",
                            "Model Confidence Score": f"{rec['predicted_rating']:.3f} ★"
                        })
                    
                    # Output table natively via Streamlit's high-speed renderer
                    st.table(formatted_data)
                    
            elif response.status_code == 422:
                st.error("❌ Validation Error: The value passed violates systemic schema bounds.")
            else:
                st.error(f"❌ Backend returned structural error code: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 **Network Connection Error!** Cannot reach the FastAPI backend.\n\n"
                "Please verify that your Uvicorn server is running actively on `http://127.0.0.1:8000` "
                "in a separate terminal window before submitting requests."
            )
        except Exception as e:
            st.error(f"An unexpected runtime exception occurred: {e}")