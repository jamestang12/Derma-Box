<?php

  //Get values passe from form in login.php
  $username = $_POST['user'];
  $password = $_POST['pass'];

  // to prevent mysql injection
  $username = stripcslashes($username);
  $password = stripcslashes($password);

  // connect to mysql
   $conn = new mysqli("34.70.40.173", "root", "password123","test") or die("Connect failed: %s\n". $conn -> error);

 // to prevent mysql injection
   $username = mysqli_real_escape_string($conn,$username);
   $password = mysqli_real_escape_string($conn, $password);
  //Query the db for user
  $result = mysqli_query($conn ,"select * from newhackdb where username = '$username' and password = '$password'") or die("Failed to query database");
  $row = mysqli_fetch_array($result);
  if ( $row['username'] == $username && $row['password'] == $password) {
    $db = mysqli_connect("34.70.40.173", "root", "password123","test");
    $mysql = "SELECT * from newhack";
    $result = $db-> query($mysql);

    echo '<body>
           <div style="overflow:hidden; background-color:lightgrey;">
             <a style="float: left; display:block; color:black; text-align: center; text-decotation: none;
             padding: 14px 16px; border solid black; font-size:17px" >'. "Dermabox" . '</a>
             <a style="float: left; display:block; color:black; text-align: center; text-decotation: none;
             padding: 14px 16px; border solid black; font-size:17px">'. "Logout" . '</a>
           </div>
         </body>';
    while ($row = $result-> fetch_assoc()) {
      // code...
//       echo $row['username']."-";
//       echo $row['password']."-";
//       echo $row['id']."-";
// 	     echo $row['data']."-";
// //

       $image = $row['image'];
       $imageData = base64_encode(file_get_contents($image));

//        echo '<img src="data:image/jpeg;base64,'.$imageData.'">';
// //
//        echo "===============" . "<br>";
       // echo "<div style='border: solid black; width: 50%; font-weight: bold;'><tr><td>" . "Name: " . $row['username'] . "</td><td><br>" .
       // "Patient Number: " . $row['id'] . "</td><td><br>" . "Result: " .
       // $row['data'] . "</td><tr><br>" . '<img onclick="window.open(\''.$image.'\', \'_blank\')" style="width:400px;
       // height:"300px;" src="data:image/jpeg;base64,'.$imageData.'">' . "</div>";


       echo '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
            <body style="background-color: gray;">
            <div style="margin-top:-120px;
                        padding: 0%;
                        color: #ffa500;
                        font-weight: bold;
                        font-size: 30px;">

              <div class="row" style="margin-top: 10%; background-color: lightgrey; border: solid black" >
                <div class="col-md-6 how-img">
                  <img onclick="window.open(\''.$image.'\', \'_blank\')"
                  src="data:image/jpeg;base64,'.$imageData.'"
                  style="width:400px; height:300px; margin-left:50px" class="rounded-circle img-fluid" alt=""/">
                  </div>
                <div class="col-md-6" style="background-color: lightgrey">
                  <h4 style="color: black;font-weight: bold; text-align: center;">'.  "Name: " . $row['username']. '</h4>
                  <hr class="hr-grey" style="margin:none;width:40%;"  >
                  <h4 style="color: black;font-weight: bold; text-align: center;">' . "Patient Id: " . $row['id'] . '</h4>
                  <h4 style="color: black;font-weight: bold; text-align: center;">' . "Result: " . $row['data'] . '</h4>
                  </div>
                  <div style="margin-left: 500px; position: absolute; color: black;">
                    <h4 style="font-weight: bold;">'. '<u>' . "Description:". '</u></h4>
                  </div>
                </div>
              </div>
            </div>
            </body>';




    }


    echo "Login sucess!! Welcome-";


  }


  else {
    echo "<script>alert('username or password incorrect')</script>";
    echo "<script>location.href='index2.php'</script>";
  }



 ?>
