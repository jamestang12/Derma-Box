<?php
  session_start();
  $count = 0;
  //connect to database
  $db = mysqli_connect("34.70.40.173", "root", "password123","test");
  if (isset($_POST['sumbit'])){
    $username = mysqli_real_escape_string($db, $_POST['user']);
    $password = mysqli_real_escape_string($db, $_POST['pass']);
    $userid = mysqli_real_escape_string($db, $_POST['id']);
    $img = "img";
    $mysql = "SELECT id from newhackdb";
    $result = $db-> query($mysql);
    if ($result-> num_rows > 0) {
      while ($row = $result-> fetch_assoc()) {
        if ($row["user"] == $username) {
          $count = 1;
          echo "<script>alert('Username already exited ')</script>";
          echo "<script>location.href='register.php'</script>";
        }
      }
    }

    if ($count == 0) {
      $sql = "INSERT INTO newhackdb(username, password,id,image) VALUES('$username','$password','$userid','$img')";
      mysqli_query($db, $sql);
      echo "<script>alert('Register successful')</script>";
      echo "<script>location.href='login.php'</script>";
    }

  }


 ?>

<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="style.css">

    <title>Register</title>
  </head>
  <body>
    <div id="frm">
      <form class="" action="register.php" method="post">
        <p>
          <label>Username ID:</label>
          <input type="text" id="id" name="id" />
        </p>
        <p>
          <label>Username:</label>
          <input type="text" id="user" name="user" />
        </p>
        <p>
          <label>Password:</label>
          <input type="password" id="pass" name="pass" />
        </p>
        <p>
          <input name="sumbit" type="submit" id="" value="submit">
        </p>


      </form>

    </div>
  </body>
</html>
