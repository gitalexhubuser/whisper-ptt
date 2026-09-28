Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = dir

localAppData = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%")

candidates = Array( _
    localAppData & "\Programs\Python\Python311\pythonw.exe", _
    localAppData & "\Programs\Python\Python312\pythonw.exe", _
    localAppData & "\Programs\Python\Python310\pythonw.exe", _
    "C:\Users\user\AppData\Local\Programs\Python\Python311\pythonw.exe", _
    "C:\Users\VisualCOde\AppData\Local\Programs\Python\Python311\pythonw.exe", _
    dir & "\.venv\Scripts\pythonw.exe" _
)

pythonw = ""
For Each path In candidates
    If fso.FileExists(path) Then
        pythonw = path
        Exit For
    End If
Next

If pythonw = "" Then
    MsgBox "pythonw.exe не найден ни по одному из стандартных путей:" & vbCrLf & _
           "- " & localAppData & "\Programs\Python\Python311\pythonw.exe" & vbCrLf & _
           "- C:\Users\VisualCOde\AppData\Local\Programs\Python\Python311\pythonw.exe" & vbCrLf & _
           "- C:\Users\user\AppData\Local\Programs\Python\Python311\pythonw.exe", 16, "Whisper PTT"
    WScript.Quit 1
End If

script = dir & "\whisper_ptt.pyw"

If Not fso.FileExists(script) Then
    MsgBox "Скрипт не найден: " & script, 16, "Whisper PTT"
    WScript.Quit 1
End If

sh.Run """" & pythonw & """ """ & script & """", 0, False