import customtkinter as ctk
from model import load_model_and_tokenizer, generate_name
from tkinter import messagebox
import random

##
## POKRETANJE GUI-a
## Kako biste pokrenuli GUI (barem zasad) odite u terminal i pozicionirajte se u src folder
## nakon toga u terminalu upisite komandu: py gui.py i dobit cete window u kojem se otvara
## mali preview izgleda naseg programa
## Isto tako make sure da imate instaliran customtkinter i tkinter da vam se uspije izgraditi
## napisan kod!
##


# sa ovom funkcijom dobivamo odabranu drzavu iz dropdowna
def get_chosen_country_index():
    return countries.index(country_var.get())

def on_click_function():
    selected_country = country_var.get()
    lang_codes = {'Croatia': 'CRO', 'Canada': 'CAN', 'Germany': 'GER', 'UK': 'UK', 'USA': 'US', 'Spain': 'SPN', 'France': 'FRA'}

    if not selected_country or selected_country == "Select a Country":
        messagebox.showwarning("Warning", "Please select a country.")
        return

    lang_code = lang_codes[selected_country]
    model, tokenizer = load_model_and_tokenizer(lang_code)
    max_seq_length = 20  # ili postavi prema duljini podataka

    generated_city_names = set() 
    while len(generated_city_names) < 25:
        name = generate_name(model, tokenizer, max_seq_length)
        generated_city_names.add(name)  

    display_city_names(list(generated_city_names)) 

def display_city_names(generated_city_names):
    for widget in city_list_frame.winfo_children():
        widget.grid_forget()

    for i, city_name in enumerate(generated_city_names):
        row = i // 5
        col = i % 5

        city_label = ctk.CTkLabel(
            master=city_list_frame,
            text=city_name,
            font=ctk.CTkFont(family="Georgia", size=14),
            text_color="#000000",
            anchor="center",
            width=80,
            height=30
        )
        city_label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

root = ctk.CTk()
root.title("City Names Generator")
root.geometry("850x570")
root.configure(fg_color="#E9EFEC")

title_label = ctk.CTkLabel(
    root,
    text="City Names Generator",
    font=ctk.CTkFont(family="Georgia", size=22,
                     weight="bold"),
    text_color="#000000",
    fg_color="#E9EFEC"
)

title_label.grid(row=0, column=0, pady=20, padx=20)

countries = ["Croatia", "Canada", "Germany", "UK", "USA", "Spain", "France"] # postavila sam redoslijed kao i u tablici za tokenizaciju
country_var = ctk.StringVar(value="Select a Country")
country_selector = ctk.CTkOptionMenu(
    root,
    variable=country_var,
    values=countries,
    font=ctk.CTkFont(family="Georgia", size=14),
    dropdown_font=ctk.CTkFont(family="Georgia", size=14),
    width=200,
    fg_color="#C4DAD2",
    text_color="#000000",
    dropdown_fg_color="#E9EFEC",
    dropdown_text_color="#000000",
    button_color="#6A9C89",
    button_hover_color="#16423C",
    dropdown_hover_color="#C4DAD2"
)
country_selector.grid(row=1, column=0, pady=15, padx=20)

confirm_button = ctk.CTkButton(
    root,
    text="Confirm Country",
    command=on_click_function,
    width=200,
    height=40,
    corner_radius=8,
    fg_color="#6A9C89",
    hover_color="#16423C",
    text_color="#E9EFEC",
)
confirm_button.grid(row=2, column=0, pady=30, padx=20)

city_list_frame = ctk.CTkFrame(root, fg_color="#E9EFEC")
city_list_frame.grid(row=3, column=0, pady=20, padx=20, sticky="nsew")

for i in range(4):
   root.grid_rowconfigure(i, weight=0)

root.grid_columnconfigure(0, weight=1)

for i in range(5):
   city_list_frame.grid_columnconfigure(
       i, weight=1)
   city_list_frame.grid_rowconfigure(
       i, weight=1)

footer_label = ctk.CTkLabel(
    root,
    text="NNProject",
    font=ctk.CTkFont(family="Georgia", size=10),
    text_color="#B0B0B0",
    fg_color="#E9EFEC"
)
footer_label.grid(row=4, column=0, pady=10, padx=20, sticky="se")

root.mainloop()
