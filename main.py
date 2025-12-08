
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from gui import SignalApp

def force_dialogs_above_root(root):
    """Monkey-patch Tkinter dialogs so all popups appear above the main window."""
    def make_parented_dialog(original_func):
        def wrapper(*args, **kwargs):
            kwargs['parent'] = root
            root.attributes('-topmost', True)
            result = original_func(*args, **kwargs)
            root.after(100, lambda: root.attributes('-topmost', False))
            return result
        return wrapper


    filedialog.askopenfilename = make_parented_dialog(filedialog.askopenfilename)
    filedialog.askopenfilenames = make_parented_dialog(filedialog.askopenfilenames)
    filedialog.asksaveasfilename = make_parented_dialog(filedialog.asksaveasfilename)
    simpledialog.askstring = make_parented_dialog(simpledialog.askstring)
    simpledialog.askfloat = make_parented_dialog(simpledialog.askfloat)
    simpledialog.askinteger = make_parented_dialog(simpledialog.askinteger)
    messagebox.showinfo = make_parented_dialog(messagebox.showinfo)
    messagebox.showerror = make_parented_dialog(messagebox.showerror)
    messagebox.showwarning = make_parented_dialog(messagebox.showwarning)
    messagebox.askquestion = make_parented_dialog(messagebox.askquestion)

if __name__ == "__main__":
    root = tk.Tk()

    app = SignalApp(root)

    force_dialogs_above_root(root)

    root.mainloop()
