<style>

.back{
  background-image: linear-gradient(to bottom, rgba(226, 226, 228, 0.52), rgba(68, 140, 182, 0.73)),
                    url('amazon_health.jpg');
  background-size: cover;
  background-repeat: no-repeat;
  height: 100vh;

}

.main-text{
  font-size: 80px;
  color: white;
  position: relative;
  top: 180px;
  margin-left: 130px;

}
.sub-text{
  text-align: left;
  font-size: 15px;
  color: white;
  font-family: Arial, Helvetica, sans-serif;
  position: relative;
  top: 200px;
  margin-left: 130px;
}

* {box-sizing: border-box;}

/* Style the navbar */
.topnav {
  overflow: hidden;
  background-color: whitesmoke;

}

/* Navbar links */
.topnav a {
  float: left;
  display: block;
  color: black;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
}

/* Navbar links on mouse-over */
.topnav a:hover {
  background-color: #ddd;
  color: black;
}

/* Active/current link */
.topnav a.active {
  background-color: #2196F3;
  color: white;
}

/* Style the input container */
.topnav .login-container {
  float: right;
}

/* Style the input field inside the navbar */
.topnav input[type=text] {
  padding: 5px;
  margin-top: 8px;
  font-size: 14px;
  border: solid lightgrey;
  width: 150px; /* adjust as needed (as long as it doesn't break the topnav) */
}

/* Style the button inside the input container */
.topnav .login-container button {
  float: right;
  padding: 6px;
  margin-top: 8px;
  margin-right: 18px;
  background: #ddd;
  font-size: 17px;
  border: none;
  border-radius: 1px;
  cursor: pointer;
}

.topnav .login-container button:hover {
  background: skyblue;
}


/* login stuff */
* {
box-sizing: border-box;
}

*:focus {
	outline: none;
}

.login {
margin-top: 100px;
margin-left: 60%;
width: 300px;
}
.login-screen {
background-color: #FFF;
padding: 20px;
border-radius: 5px;
}

.app-title {
color: #777;
}

.login-form {
text-align: center;
}
.control-group {
margin-bottom: 10px;
}

input {
text-align: center;
background-color: #ECF0F1;
border: 2px solid transparent;
border-radius: 3px;
font-size: 16px;
font-weight: 200;
padding: 10px 0;
width: 250px;
transition: border .5s;
}

input:focus {
border: 2px solid #3498DB;
box-shadow: none;
}

.btn {
  border: 2px solid transparent;
  background: #3498DB;
  color: #ffffff;
  font-size: 16px;
  line-height: 25px;
  padding: 10px 20px;
  text-decoration: none;
  text-shadow: none;
  border-radius: 3px;
  box-shadow: none;
  transition: 0.25s;
  display: block;
  width: 250px;
  margin: 0 auto;
}

.btn:hover {
  background-color: #2980B9;
}

.login-link {
  font-size: 12px;
  color: #444;
  display: block;
  margin-top: 12px;
}

/* Add responsiveness - On small screens, display the navbar vertically instead of horizontally */
@media screen and (max-width: 600px) {
  .topnav .login-container {
    float: none;
  }
  .topnav a, .topnav input[type=text], .topnav .login-container button {
    float: none;
    display: flex;
    text-align: left;
    width: 100%;
    margin: 0;
    padding: 14px;
  }



}

</style>

<html>
    <body>

    <div class="topnav">
        <a class="active" href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#contact">Contact</a>
        <!-- <div class="login-container">
            <form action="/action_page.php">
            <input type="text" placeholder="Username" name="username"></input>
            <input type="text" placeholder="Password" name="psw"></input>
            <button type="submit">Login</button>
            </form>  -->
        </div>
    </div>

    </body>
    <body>
        <div class="back">
            <strong class="main-text">AleshiaBox2.0 </strong>
            <h1 class="sub-text">Amazing new product we yoloed</h1>





            <div class="login">
                <div class="login-screen">
                    <div class="app-title">
                        <h1>Login</h1>
                    </div>

                    <div class="login-form">

                        <div class="control-group">
                          <form class="" action="process.php" method="post">
                            <input type="text" class="login-field"  placeholder="username" id="user" name="user">
                          <label class="login-field-icon fui-user" for="login-name"></label>
                          </div>

                          <div class="control-group">
                            <input type="password" class="login-field"  placeholder="password" id="pass" name="pass">
                          <label class="login-field-icon fui-lock" for="login-pass"></label>
                          </div>

                          <input class="btn btn-primary btn-large btn-block" type="submit" id="" value="Login">
                          </form>


                    </div>
                </div>
            </div>
        </div>

    </body>
</html>
