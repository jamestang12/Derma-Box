
<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">
    <title>Login Page</title>
    <link rel="stylesheet" href="style.css">
  </head>
  <body>
    <div id="frm">
      <form class="" action="process.php" method="post">
        <p>
          <label>Username:</label>
          <input type="text" id="user" name="user" />
        </p>
        <p>
          <label>Password:</label>
          <input type="password" id="pass" name="pass" />
        </p>
        <p>
          <input type="submit" id="" value="Login">
          <button onclick="window.location.href = 'register.php'" type="button" name="Register">Register</button>
        </p>


      </form>

    </div>

  </body>
</html>
