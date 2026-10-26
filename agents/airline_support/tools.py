"""The 17 tools the airline support bot can call.

Ported from the open source LangGraph "Build a Customer Support Bot".
SQL, return messages and docstrings are kept as in the original: the docstrings
are what the model reads to choose a tool, and we are testing the real agent,
not an improved version of it. Known weaknesses (e.g. no ownership check on
hotel/car/excursion tools) are kept on purpose and documented in ADDITIONS.md.

Changes from the tutorial, only what local + deterministic running needs:
- build_tools() takes the DB path instead of a global `db`, so every test case
  can run against its own fresh copy of the database.
- update_ticket_to_new_flight uses the frozen "now" instead of the real clock.
- lookup_policy uses the local Qwen PolicyRetriever instead of OpenAI embeddings.
"""
import sqlite3
from datetime import date, datetime
from typing import Optional, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .policy import PolicyRetriever


def build_tools(db: str, frozen_now: datetime, retriever: PolicyRetriever, top_k: int = 2) -> list:
    """Create the agent's tools, bound to one database file and one frozen "now"."""

    @tool
    def lookup_policy(query: str) -> str:
        """Consult the company policies to check whether certain options are permitted.
        Use this before making any flight changes performing other 'write' events."""
        docs = retriever.query(query, k=top_k)
        return "\n\n".join([doc["section"] for doc in docs])

    @tool
    def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
        """Fetch all tickets for the user along with corresponding flight information and seat assignments.

        Returns:
            A list of dictionaries where each dictionary contains the ticket details,
            associated flight details, and the seat assignments for each ticket belonging to the user.
        """
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            raise ValueError("No passenger ID configured.")

        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = """
        SELECT
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
            bp.seat_no, tf.fare_conditions
        FROM
            tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
        WHERE
            t.passenger_id = ?
        """
        cursor.execute(query, (passenger_id,))
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        conn.close()

        return results

    @tool
    def search_flights(
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        start_time: Optional[date | datetime] = None,
        end_time: Optional[date | datetime] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search for flights based on departure airport, arrival airport, and departure time range."""
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = "SELECT * FROM flights WHERE 1 = 1"
        params = []

        if departure_airport:
            query += " AND departure_airport = ?"
            params.append(departure_airport)

        if arrival_airport:
            query += " AND arrival_airport = ?"
            params.append(arrival_airport)

        if start_time:
            query += " AND scheduled_departure >= ?"
            params.append(start_time)

        if end_time:
            query += " AND scheduled_departure <= ?"
            params.append(end_time)
        query += " LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        conn.close()

        return results

    @tool
    def update_ticket_to_new_flight(
        ticket_no: str, new_flight_id: int, *, config: RunnableConfig
    ) -> str:
        """Update the user's ticket to a new valid flight."""
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            raise ValueError("No passenger ID configured.")

        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
            (new_flight_id,),
        )
        new_flight = cursor.fetchone()
        if not new_flight:
            cursor.close()
            conn.close()
            return "Invalid new flight ID provided."
        column_names = [column[0] for column in cursor.description]
        new_flight_dict = dict(zip(column_names, new_flight))
        # Tutorial: datetime.now(tz=pytz.timezone("Etc/GMT-3")). Frozen "now" instead (Principle 1).
        current_time = frozen_now
        departure_time = datetime.strptime(
            new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
        )
        time_until = (departure_time - current_time).total_seconds()
        if time_until < (3 * 3600):
            return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        current_flight = cursor.fetchone()
        if not current_flight:
            cursor.close()
            conn.close()
            return "No existing ticket found for the given ticket number."

        # Check the signed-in user actually has this ticket
        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            cursor.close()
            conn.close()
            return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

        cursor.execute(
            "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
            (new_flight_id, ticket_no),
        )
        conn.commit()

        cursor.close()
        conn.close()
        return "Ticket successfully updated to new flight."

    @tool
    def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
        """Cancel the user's ticket and remove it from the database."""
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            raise ValueError("No passenger ID configured.")
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        existing_ticket = cursor.fetchone()
        if not existing_ticket:
            cursor.close()
            conn.close()
            return "No existing ticket found for the given ticket number."

        # Check the signed-in user actually has this ticket
        cursor.execute(
            "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            cursor.close()
            conn.close()
            return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

        cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
        conn.commit()

        cursor.close()
        conn.close()
        return "Ticket successfully cancelled."

    @tool
    def search_car_rentals(
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
    ) -> list[dict]:
        """
        Search for car rentals based on location, name, price tier, start date, and end date.

        Args:
            location (Optional[str]): The location of the car rental. Defaults to None.
            name (Optional[str]): The name of the car rental company. Defaults to None.
            price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
            start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

        Returns:
            list[dict]: A list of car rental dictionaries matching the search criteria.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = "SELECT * FROM car_rentals WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
        cursor.execute(query, params)
        results = cursor.fetchall()

        conn.close()

        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in results
        ]

    @tool
    def book_car_rental(rental_id: int) -> str:
        """
        Book a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to book.

        Returns:
            str: A message indicating whether the car rental was successfully booked or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully booked."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."

    @tool
    def update_car_rental(
        rental_id: int,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
    ) -> str:
        """
        Update a car rental's start and end dates by its ID.

        Args:
            rental_id (int): The ID of the car rental to update.
            start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

        Returns:
            str: A message indicating whether the car rental was successfully updated or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date, rental_id),
            )
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully updated."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."

    @tool
    def cancel_car_rental(rental_id: int) -> str:
        """
        Cancel a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to cancel.

        Returns:
            str: A message indicating whether the car rental was successfully cancelled or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully cancelled."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."

    @tool
    def search_hotels(
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
    ) -> list[dict]:
        """
        Search for hotels based on location, name, price tier, check-in date, and check-out date.

        Args:
            location (Optional[str]): The location of the hotel. Defaults to None.
            name (Optional[str]): The name of the hotel. Defaults to None.
            price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
            checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

        Returns:
            list[dict]: A list of hotel dictionaries matching the search criteria.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = "SELECT * FROM hotels WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For the sake of this tutorial, we will let you match on any dates and price tier.
        cursor.execute(query, params)
        results = cursor.fetchall()

        conn.close()

        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in results
        ]

    @tool
    def book_hotel(hotel_id: int) -> str:
        """
        Book a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to book.

        Returns:
            str: A message indicating whether the hotel was successfully booked or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully booked."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @tool
    def update_hotel(
        hotel_id: int,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
    ) -> str:
        """
        Update a hotel's check-in and check-out dates by its ID.

        Args:
            hotel_id (int): The ID of the hotel to update.
            checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

        Returns:
            str: A message indicating whether the hotel was successfully updated or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        if checkin_date:
            cursor.execute(
                "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
            )
        if checkout_date:
            cursor.execute(
                "UPDATE hotels SET checkout_date = ? WHERE id = ?",
                (checkout_date, hotel_id),
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully updated."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @tool
    def cancel_hotel(hotel_id: int) -> str:
        """
        Cancel a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to cancel.

        Returns:
            str: A message indicating whether the hotel was successfully cancelled or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully cancelled."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."

    @tool
    def search_trip_recommendations(
        location: Optional[str] = None,
        name: Optional[str] = None,
        keywords: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for trip recommendations based on location, name, and keywords.

        Args:
            location (Optional[str]): The location of the trip recommendation. Defaults to None.
            name (Optional[str]): The name of the trip recommendation. Defaults to None.
            keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

        Returns:
            list[dict]: A list of trip recommendation dictionaries matching the search criteria.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = "SELECT * FROM trip_recommendations WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if keywords:
            keyword_list = keywords.split(",")
            keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
            query += f" AND ({keyword_conditions})"
            params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

        cursor.execute(query, params)
        results = cursor.fetchall()

        conn.close()

        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in results
        ]

    @tool
    def book_excursion(recommendation_id: int) -> str:
        """
        Book a excursion by its recommendation ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to book.

        Returns:
            str: A message indicating whether the trip recommendation was successfully booked or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Trip recommendation {recommendation_id} successfully booked."
        else:
            conn.close()
            return f"No trip recommendation found with ID {recommendation_id}."

    @tool
    def update_excursion(recommendation_id: int, details: str) -> str:
        """
        Update a trip recommendation's details by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to update.
            details (str): The new details of the trip recommendation.

        Returns:
            str: A message indicating whether the trip recommendation was successfully updated or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET details = ? WHERE id = ?",
            (details, recommendation_id),
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Trip recommendation {recommendation_id} successfully updated."
        else:
            conn.close()
            return f"No trip recommendation found with ID {recommendation_id}."

    @tool
    def cancel_excursion(recommendation_id: int) -> str:
        """
        Cancel a trip recommendation by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to cancel.

        Returns:
            str: A message indicating whether the trip recommendation was successfully cancelled or not.
        """
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Trip recommendation {recommendation_id} successfully cancelled."
        else:
            conn.close()
            return f"No trip recommendation found with ID {recommendation_id}."

    return [
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        update_ticket_to_new_flight,
        cancel_ticket,
        search_car_rentals,
        book_car_rental,
        update_car_rental,
        cancel_car_rental,
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
        search_trip_recommendations,
        book_excursion,
        update_excursion,
        cancel_excursion,
    ]
