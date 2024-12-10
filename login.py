import hashlib
from hashlib import sha256
from time import sleep

import streamlit as st


def is_logged_in(u_session_state):
    return (
        u_session_state["authenticated"]
        if "authenticated" in u_session_state
        else False
    )


def login(u_session_state):
    if is_logged_in(u_session_state):
        return

    # Placeholder for password
    password = st.text_input("Password", type="password")
    hashed_password = hashlib.sha256(password.encode("utf-8")).hexdigest()

    # Define your password here
    correct_password_hashed = (
        '8db0f4293d7f8590d4c05cb1eee6b7916f576da0b27bf24c3c634a0e151c2084'
    )

    if not password or password == "":
        st.info("Please enter a password")
        st.stop()

    if hashed_password == correct_password_hashed:
        u_session_state["authenticated"] = True
        st.success("You are now logged in!")
        sleep(2)
        st.rerun()

    if not is_logged_in(u_session_state):
        st.error("The password you entered is incorrect")
        st.stop()
