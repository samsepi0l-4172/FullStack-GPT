# Import the streamlit library. Streamlit is a framework for building machine learning and data science web apps.
import streamlit as st

# Set the configuration for the page using the `set_page_config` method.
# `page_title` sets the title of the page that appears in the browser tab.
# `page_icon` sets the favicon of the page, which appears in the browser tab next to the title.
st.set_page_config(
    page_title="FullstackGPT Home",  # The title of the page
    page_icon="🤖",  # The icon of the page
)

# Use the `markdown` method to create a markdown section on the web page.
# The markdown text includes a welcome message and a list of apps.
# The list items are links to different pages of the web app.
st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)  # Link to the DocumentGPT app
- [ ] [PrivateGPT](/PrivateGPT)  # Link to the PrivateGPT app
- [ ] [QuizGPT](/QuizGPT)  # Link to the QuizGPT app
- [ ] [SiteGPT](/SiteGPT)  # Link to the SiteGPT app
- [ ] [MeetingGPT](/MeetingGPT)  # Link to the MeetingGPT app
- [ ] [InvestorGPT](/InvestorGPT)  # Link to the InvestorGPT app
    """
)