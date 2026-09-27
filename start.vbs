Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = dir

pythonw = "C:\Users\VisualCode\AppData\Local\Programs\Python\Python311\pythonw.exe"
script  = dir & "\whisper_ptt.pyw"

If Not fso.FileExists(pythonw) Then
    MsgBox "pythonw.exe не найден: " & pythonw, 16, "Whisper PTT"
    WScript.Quit 1
End If

If Not fso.FileExists(script) Then
    MsgBox "Скрипт не найден: " & script, 16, "Whisper PTT"
    WScript.Quit 1
End If

sh.Run """" & pythonw & """ """ & script & """", 0, False